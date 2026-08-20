"""Named episode-generation providers for imitation dataset collection."""

from __future__ import annotations

from typing import Any, Callable

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic import VoxelReplanningAStarHeuristic, build_voxel_heuristic


class GenerationProviderError(ValueError):
    """Raised when a generation provider name is unknown or misbehaves."""


class EpisodeGenerationContext(BaseModel):
    """Live inputs handed to an episode-generation provider for one attempt.

    ``env`` has already been reset for this attempt; ``observation`` is the
    raw observation ``env.reset`` returned. The provider must not apply RLlib
    preprocessing -- the caller does that once the episode is returned.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: VoxelEnv
    observation: Any
    seed: int
    attempt: int
    parameters: dict[str, Any] = Field(default_factory=dict)


class DemonstrationEpisode(BaseModel):
    """One recorded rollout produced by a generation provider."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observations: list[Any]
    actions: list[Any]
    success: bool
    seed: int


GenerationProvider = Callable[[EpisodeGenerationContext], DemonstrationEpisode]


def _generate_heuristic_episode(
    context: EpisodeGenerationContext,
    heuristic_type: str,
) -> DemonstrationEpisode:
    """Roll out a `heuristic/voxel/*` teacher, recording raw observations.

    Replicates the per-step loop `collect_demonstrations` used to run inline
    (compute a fixed action plan, or replan every step for
    `replanning_astar`), so built-in provider behavior is unchanged from
    before this refactor.
    """
    weight = context.parameters.get("weight")
    if heuristic_type == "weighted_astar":
        if weight is not None and weight <= 0.0:
            raise GenerationProviderError(
                "generation provider 'weighted_astar' requires parameters.weight > 0"
            )
    elif weight is not None:
        raise GenerationProviderError(
            f"generation provider {heuristic_type!r} does not accept parameters.weight"
        )
    env = context.env
    teacher = build_voxel_heuristic(env, heuristic_type, weight=weight)

    observation = context.observation
    observations: list[Any] = []
    actions: list[Any] = []
    success = False

    try:
        if isinstance(teacher, VoxelReplanningAStarHeuristic):
            action_plan = None
        else:
            action_plan = list(teacher.plan().action_indices)

        step_index = 0
        while True:
            if action_plan is None:
                current_plan = teacher.plan()
                if not current_plan.action_indices:
                    break
                action = current_plan.action_indices[0]
            else:
                if step_index >= len(action_plan):
                    break
                action = action_plan[step_index]

            observations.append(observation)
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
            step_index += 1
            success = bool(info.get("goal_reached", False))
            if terminated or truncated:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        success = False

    return DemonstrationEpisode(
        observations=observations, actions=actions, success=success, seed=context.seed
    )


BUILT_IN_GENERATION_PROVIDERS: dict[str, GenerationProvider] = {
    "astar": lambda context: _generate_heuristic_episode(context, "astar"),
    "dijkstra": lambda context: _generate_heuristic_episode(context, "dijkstra"),
    "weighted_astar": lambda context: _generate_heuristic_episode(context, "weighted_astar"),
    "replanning_astar": lambda context: _generate_heuristic_episode(
        context, "replanning_astar"
    ),
}


def resolve_generation_provider(
    name: str,
    *,
    python_provider: Any | None = None,
) -> GenerationProvider:
    """Resolve a generation provider by name, built-in first, then Python."""
    built_in = BUILT_IN_GENERATION_PROVIDERS.get(name)
    if built_in is not None:
        return built_in
    if python_provider is not None and getattr(python_provider, "name", None) == name:
        return python_provider.generate
    raise GenerationProviderError(f"unknown generation provider: {name!r}")
