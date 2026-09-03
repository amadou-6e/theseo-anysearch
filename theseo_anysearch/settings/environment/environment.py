"""Composed environment settings and runtime translation."""

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from theseo_anysearch.environments.task import TaskConfig
from theseo_anysearch.settings.compatibility import NestedFieldAccessMixin
from theseo_anysearch.settings.environment.action import ActionConfig
from theseo_anysearch.settings.environment.agent import (
    AgentConfig,
    HunterAndHuntedConfig,
)
from theseo_anysearch.settings.environment.curriculum import WaypointCurriculumConfig
from theseo_anysearch.settings.environment.geometry import GeometryConfig
from theseo_anysearch.settings.environment.lifecycle import LifecycleConfig
from theseo_anysearch.settings.environment.observation import ObservationConfig
from theseo_anysearch.settings.environment.rewards import RewardConfig
from theseo_anysearch.settings.environment.scenarios import ScenarioConfig

_LEGACY_ENV_FIELDS: dict[str, tuple[str, str]] = {
    "stl_path": ("geometry", "stl_path"),
    "stl_paths": ("geometry", "stl_paths"),
    "scale": ("geometry", "scale"),
    "scale_range": ("geometry", "scale_range"),
    "grid_size": ("geometry", "grid_size"),
    "extent": ("geometry", "extent"),
    "geometry_boxes": ("geometry", "boxes"),
    "geometry_pool_size": ("geometry", "pool_size"),
    "scale_variants_per_map": ("geometry", "scale_variants_per_map"),
    "geometry_padding": ("geometry", "padding"),
    "geometry_pool": ("geometry", "pool"),
    "obs_mode": ("observation", "mode"),
    "box_radius": ("observation", "box_radius"),
    "box_radii": ("observation", "box_radii"),
    "ray_max_len": ("observation", "ray_max_len"),
    "action_mode": ("action", "mode"),
    "step_cost": ("rewards", "step_cost"),
    "collision_cost": ("rewards", "collision_cost"),
    "goal_reward": ("rewards", "goal_reward"),
    "distance_shaping": ("rewards", "distance_shaping"),
    "distance_reward_mode": ("rewards", "distance_reward_mode"),
    "zone_reward_min": ("rewards", "zone_reward_min"),
    "zone_reward_max": ("rewards", "zone_reward_max"),
    "zone_reward_curve": ("rewards", "zone_reward_curve"),
    "distance_metric": ("rewards", "distance_metric"),
    "invalid_action_cost": ("rewards", "invalid_action_cost"),
    "construction_residual_weight": ("rewards", "construction_residual_weight"),
    "construction_overshoot_weight": ("rewards", "construction_overshoot_weight"),
}


class EnvConfig(NestedFieldAccessMixin, BaseModel):
    """Environment settings grouped by geometry, observation, action, and rewards."""

    model_config = ConfigDict(extra="forbid")
    exposed_nested_fields: ClassVar[tuple[str, ...]] = (
        "geometry",
        "observation",
        "action",
        "rewards",
    )
    agent_count: int = Field(default=4, ge=1, description="Number of agents in the environment.")
    max_steps: int = Field(default=200, ge=1, description="Maximum actions permitted in one episode.")
    seed: int = Field(42, description="Environment random seed.")
    trail_mode: bool = Field(True, description="Fill each entered voxel for legacy trail behavior.")
    target_fill: int | None = Field(default=None, ge=0, description="Optional target number of agent-filled voxels.")
    waypoints_file: str | None = Field(None, description="Path to fixed start and goal waypoints.")
    waypoint_curriculum: WaypointCurriculumConfig = Field(
        default_factory=WaypointCurriculumConfig,
        description="Progressive waypoint or segmented-route curriculum.",
    )
    task: TaskConfig = Field(default_factory=TaskConfig, description="Task success and termination contract.")
    geometry: GeometryConfig = Field(default_factory=GeometryConfig, description="Geometry and voxel-grid settings.")
    observation: ObservationConfig = Field(default_factory=ObservationConfig, description="Policy observation settings.")
    action: ActionConfig = Field(default_factory=ActionConfig, description="Policy action-space settings.")
    rewards: RewardConfig = Field(default_factory=RewardConfig, description="Reward terms and provider selection.")
    lifecycle: LifecycleConfig = Field(
        default_factory=LifecycleConfig,
        description="Ordered rules that resolve episode outcomes and diagnostics.",
    )
    scenarios: ScenarioConfig = Field(default_factory=ScenarioConfig, description="Episode scenario provider.")
    agents: tuple[AgentConfig, ...] | None = Field(
        None, description="Ordered per-agent settings for heterogeneous environments."
    )
    hunter_and_hunted: HunterAndHuntedConfig | None = Field(
        None, description="Optional asymmetric capture task."
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_blocks(cls, value: Any) -> Any:
        """Accept legacy-only input during migration, but reject mixed blocks."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("agents") is not None and "agent_count" not in data:
            data["agent_count"] = len(data["agents"])
        legacy_by_block: dict[str, list[str]] = {}
        for legacy, (block, _) in _LEGACY_ENV_FIELDS.items():
            if legacy in data:
                legacy_by_block.setdefault(block, []).append(legacy)
        conflicts = sorted(block for block in legacy_by_block if block in data)
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(
                f"legacy flattened environment fields cannot be mixed with nested blocks: {joined}"
            )
        for block, legacy_fields in legacy_by_block.items():
            nested: dict[str, Any] = {}
            for legacy in legacy_fields:
                _, nested_name = _LEGACY_ENV_FIELDS[legacy]
                nested[nested_name] = data.pop(legacy)
            data[block] = nested
        return data

    @model_validator(mode="after")
    def validate_heterogeneous_agents(self) -> "EnvConfig":
        """Validate agent identity, count, and capture-task references."""
        if self.agents is None:
            if self.hunter_and_hunted is not None:
                raise ValueError("hunter_and_hunted requires env.agents")
            return self
        if len(self.agents) != self.agent_count:
            raise ValueError("agent_count must equal the number of env.agents")
        names = [agent.id for agent in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("env.agents ids must be unique")
        if self.hunter_and_hunted is not None:
            configured = set(names)
            if self.hunter_and_hunted.hunter not in configured:
                raise ValueError("hunter_and_hunted.hunter must name a configured agent")
            if self.hunter_and_hunted.hunted not in configured:
                raise ValueError("hunter_and_hunted.hunted must name a configured agent")
            if self.hunter_and_hunted.hunter == self.hunter_and_hunted.hunted:
                raise ValueError("hunter and hunted must be different agents")
        return self

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return the flat dictionary consumed by the existing environments."""
        action_predicates, action_outcomes = self.action.resolved_pipeline(
            trail_mode=self.trail_mode
        )
        heterogeneous_agents = None
        if self.agents is not None:
            heterogeneous_agents = []
            for agent in self.agents:
                predicates, outcomes = agent.action.resolved_pipeline(trail_mode=False)
                heterogeneous_agents.append(
                    {
                        "id": agent.id,
                        "policy": agent.policy or agent.id,
                        "action_mode": agent.action.mode,
                        "start": list(agent.start) if agent.start is not None else None,
                        "action_predicates": [item.model_dump(mode="json") for item in predicates],
                        "action_outcomes": [item.model_dump(mode="json") for item in outcomes],
                        "action_history_length": agent.action.history_length,
                    }
                )
        return {
            "geometry_provider": (
                self.geometry.provider.name if self.geometry.provider else None
            ),
            "geometry_provider_parameters": (
                self.geometry.provider.parameters if self.geometry.provider else {}
            ),
            "geometry_sources": [
                source.model_dump(mode="json") for source in self.geometry.sources
            ],
            "stl_path": str(self.geometry__stl_path) if self.geometry__stl_path else None,
            "stl_paths": (
                [str(path) for path in self.geometry__stl_paths] if self.geometry__stl_paths else None
            ),
            "scale": self.geometry__scale,
            "scale_range": self.geometry__scale_range,
            "grid_size": self.geometry__grid_size,
            "extent": self.geometry.extent,
            "geometry_boxes": self.geometry__boxes,
            "geometry_pool_size": self.geometry__pool_size,
            "scale_variants_per_map": self.geometry__scale_variants_per_map,
            "geometry_padding": self.geometry__padding,
            "geometry_pool": self.geometry__pool,
            "geometry_validation": self.geometry.validation.model_dump(mode="json"),
            "compiled_world_path": (
                str(self.geometry.compiled_world_path)
                if self.geometry.compiled_world_path is not None
                else None
            ),
            "world_maximum_decoded_bytes": self.geometry.maximum_decoded_bytes,
            "world_prefetch_margin": self.geometry.prefetch_margin,
            "obs_mode": self.observation__mode,
            "box_radius": self.observation__box_radius,
            "box_radii": self.observation__box_radii,
            "ray_max_len": self.observation__ray_max_len,
            "action_mode": self.action__mode,
            "action_behavior": self.action.behavior,
            "action_predicates": [
                item.model_dump(mode="json") for item in action_predicates
            ],
            "action_outcomes": [
                item.model_dump(mode="json") for item in action_outcomes
            ],
            "action_history_length": self.action.history_length,
            "action_masking_enabled": self.action.masking.enabled,
            "action_masking_all_masked": self.action.masking.all_masked,
            "agent_count": self.agent_count,
            "agents": heterogeneous_agents,
            "hunter_and_hunted": (
                self.hunter_and_hunted.model_dump(mode="json")
                if self.hunter_and_hunted is not None
                else None
            ),
            "max_steps": self.max_steps,
            "seed": self.seed,
            "trail_mode": self.trail_mode,
            "target_fill": self.target_fill,
            "waypoints_file": self.waypoints_file,
            "waypoint_curriculum": self.waypoint_curriculum.model_dump(mode="json"),
            "step_cost": self.rewards__step_cost,
            "collision_cost": self.rewards__collision_cost,
            "goal_reward": self.rewards__goal_reward,
            "distance_shaping": self.rewards__distance_shaping,
            "distance_reward_mode": self.rewards__distance_reward_mode,
            "zone_reward_min": self.rewards__zone_reward_min,
            "zone_reward_max": self.rewards__zone_reward_max,
            "zone_reward_curve": self.rewards__zone_reward_curve,
            "distance_metric": self.rewards__distance_metric,
            "invalid_action_cost": self.rewards__invalid_action_cost,
            "construction_residual_weight": self.rewards__construction_residual_weight,
            "construction_overshoot_weight": self.rewards__construction_overshoot_weight,
            "custom_reward": self.rewards__provider.name if self.rewards__provider else None,
            "custom_reward_parameters": (
                self.rewards__provider.parameters if self.rewards__provider else {}
            ),
            "scenario_provider": self.scenarios.provider.name if self.scenarios.provider else None,
            "scenario_parameters": self.scenarios.provider.parameters if self.scenarios.provider else {},
            "scenario_candidate_index": (
                str(self.scenarios.candidate_index)
                if self.scenarios.candidate_index is not None
                else None
            ),
            "scenario_maximum_candidate_queries": self.scenarios.maximum_candidate_queries,
            "scenario_maximum_candidate_results": self.scenarios.maximum_candidate_results,
            "task": self.task.model_dump(mode="json"),
            "lifecycle_rules": [
                rule.model_dump(mode="json") for rule in self.lifecycle.rules
            ],
        }
