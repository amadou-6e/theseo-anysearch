"""Convention-based custom reward discovery and execution."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomRewardError(ValueError):
    """Raised when a custom reward module violates the reward contract."""


class RewardContext(BaseModel):
    """Immutable inputs provided to ``compute_reward(context)`` each step."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    step: int
    action: Any
    action_index: int
    previous_observation: Any
    observation: Any
    previous_cursor: tuple[int, int, int]
    cursor: tuple[int, int, int]
    goal: tuple[int, int, int] | None
    previous_goal_distance: float
    goal_distance: float
    invalid_action: bool
    collision: bool
    terminated: bool
    truncated: bool
    standard_reward: float
    standard_breakdown: dict[str, float]
    env_config: dict[str, Any]
    info: dict[str, Any] = Field(default_factory=dict)


class RewardResult(BaseModel):
    """Custom reward value and its additive or replacement semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reward: float
    components: dict[str, float] = Field(default_factory=dict)
    mode: Literal["add", "replace"] = "add"

    @model_validator(mode="after")
    def validate_reward(self) -> "RewardResult":
        if not math.isfinite(self.reward):
            raise ValueError("custom reward must be finite")
        for name, value in self.components.items():
            if not name.isidentifier():
                raise ValueError("custom reward component names must be Python identifiers")
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"custom reward component {name!r} must be finite")
        if self.components and not math.isclose(
            sum(self.components.values()),
            self.reward,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("custom reward components must sum to reward")
        return self


RewardFunction = Callable[[RewardContext], RewardResult]


class RewardProvider(BaseModel):
    """A validated custom reward function and its source identity."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    source_path: Path
    source_sha256: str
    compute_reward: RewardFunction = Field(exclude=True)


def discover_reward_source(config_path: Path | None, reward_name: str | None) -> Path | None:
    """Discover the shared ``rewards.py`` module for a named reward."""
    if config_path is None or reward_name is None:
        return None
    source = config_path.with_name("rewards.py")
    return source if source.is_file() else None


def copy_reward_source(
    config_path: Path | None,
    destination: Path,
    reward_name: str | None,
) -> Path | None:
    """Archive the named reward module under the stable name ``rewards.py``."""
    source = discover_reward_source(config_path, reward_name)
    if source is None:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination.joinpath("rewards.py")
    shutil.copy2(source, target)
    return target


def load_reward_provider(
    source_path: Path | None,
    reward_name: str | None,
) -> RewardProvider | None:
    """Import one YAML-selected function from an optional ``rewards.py`` module."""
    if source_path is None or reward_name is None:
        return None
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"_theseo_anysearch_rewards_{digest[:16]}", source_path
    )
    if spec is None or spec.loader is None:
        raise CustomRewardError(f"Cannot import custom rewards from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, reward_name, None)
    if not callable(function):
        raise CustomRewardError(
            f"{source_path} must define callable {reward_name}(context)"
        )
    if len(inspect.signature(function).parameters) != 1:
        raise CustomRewardError(
            f"{source_path}: {reward_name} must accept exactly one argument"
        )
    return RewardProvider(
        name=reward_name,
        source_path=source_path,
        source_sha256=digest,
        compute_reward=function,
    )
def apply_custom_reward(
    provider: RewardProvider | None,
    context: RewardContext,
) -> tuple[float, dict[str, float]]:
    """Apply one validated custom result to the built-in reward breakdown."""
    if provider is None:
        return context.standard_reward, dict(context.standard_breakdown)
    try:
        result = RewardResult.model_validate(provider.compute_reward(context))
    except Exception as exc:
        raise CustomRewardError(
            f"{provider.source_path}: invalid custom reward result: {exc}"
        ) from exc

    component_name = provider.name
    components = dict(result.components) or {component_name: result.reward}
    collisions = set(components) & set(context.standard_breakdown)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise CustomRewardError(
            f"{provider.source_path}: custom components collide with built-ins: {names}"
        )
    if result.mode == "replace":
        return result.reward, components
    return (
        context.standard_reward + result.reward,
        {**context.standard_breakdown, **components},
    )


def write_reward_manifest(
    provider: RewardProvider | None,
    destination: Path,
) -> Path | None:
    """Record the archived reward source and hash in run artifacts."""
    if provider is None:
        return None
    path = destination.joinpath("custom_reward.json")
    path.write_text(
        json.dumps(
            {
                "name": provider.name,
                "source": provider.source_path.name,
                "sha256": provider.source_sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
