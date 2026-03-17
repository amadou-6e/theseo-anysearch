import pytest

from theseo_anysearch.rllib.rewards.base import RewardShaper
from theseo_anysearch.rllib.rewards.models import RewardConfig, CompositeRewardConfig


class PassthroughShaper(RewardShaper):
    """Concrete shaper that returns reward unchanged."""
    def shape(self, agent_id: str, reward: float, obs: dict, done: bool) -> float:
        return reward


class ScaledShaper(RewardShaper):
    """Concrete shaper that multiplies reward by a scale factor."""
    def __init__(self, scale: float) -> None:
        self._scale = scale

    def shape(self, agent_id: str, reward: float, obs: dict, done: bool) -> float:
        return reward * self._scale


class TestRewardShaper:
    def test_passthrough_returns_same_reward(self):
        shaper = PassthroughShaper()
        assert shaper.shape("agent_0", 1.5, {}, False) == 1.5

    def test_passthrough_handles_negative(self):
        shaper = PassthroughShaper()
        assert shaper.shape("agent_0", -0.3, {}, False) == pytest.approx(-0.3)

    def test_scaled_shaper_multiplies(self):
        shaper = ScaledShaper(2.0)
        assert shaper.shape("agent_0", 3.0, {}, False) == pytest.approx(6.0)

    def test_done_flag_forwarded(self):
        calls = []
        class RecordingShaper(RewardShaper):
            def shape(self, agent_id, reward, obs, done):
                calls.append(done)
                return reward
        shaper = RecordingShaper()
        shaper.shape("a", 1.0, {}, True)
        assert calls == [True]


class TestRewardConfig:
    def test_default_weight(self):
        cfg = RewardConfig()
        assert cfg.weight == 1.0

    def test_custom_weight(self):
        cfg = RewardConfig(weight=0.5)
        assert cfg.weight == pytest.approx(0.5)


class TestCompositeRewardConfig:
    def test_empty_shapers(self):
        cfg = CompositeRewardConfig()
        assert cfg.shapers == []
