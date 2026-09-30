"""Unit tests for reward zone environment configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from theseo_anysearch.models import EnvConfig


class TestRewardZoneConfig:
    """Validate dense distance-zone reward config fields."""

    def test_accepts_zone_reward_mode(self):
        cfg = EnvConfig(
            distance_reward_mode="zone",
            zone_reward_min=-2.0,
            zone_reward_max=-0.05,
            zone_reward_curve="exponential",
        )

        assert cfg.rewards__distance_reward_mode == "zone"
        assert cfg.rewards__zone_reward_curve == "exponential"

    def test_rejects_non_negative_zone_max(self):
        with pytest.raises(ValidationError, match="zone_reward_max must stay negative"):
            EnvConfig(distance_reward_mode="zone", zone_reward_max=0.0)

    def test_rejects_inverted_zone_range(self):
        with pytest.raises(
            ValidationError,
            match="zone_reward_min must be less than or equal to zone_reward_max",
        ):
            EnvConfig(
                distance_reward_mode="zone",
                zone_reward_min=-0.01,
                zone_reward_max=-1.0,
            )
