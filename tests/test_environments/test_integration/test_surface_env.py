"""
Integration tests for SurfaceEnv — require theseo_core wheel.
Run with: pytest tests/integration/ -m integration
"""
import numpy as np
import pytest

theseo_core = pytest.importorskip("theseo_core", reason="theseo_core wheel not installed")

from theseo_anysearch.environments.pettingzoo.surface_env import SurfaceEnv  # noqa: E402


# 20-coord line surface: plenty of room for 2 agents (needs 4 distinct coords).
LINE_SURFACE = [(i, 0, 0) for i in range(20)]

SURFACE_CONFIG = {
    "agent_count": 2,
    "max_steps": 50,
    "seed": 7,
}


@pytest.fixture()
def env():
    """Provide env."""
    return SurfaceEnv(SURFACE_CONFIG)


@pytest.mark.integration
class TestSurfaceEnvIntegration:
    """Tests SurfaceEnvIntegration."""
    def test_rust_env_is_not_none(self, env):
        """_build_rust_env must return a live theseo_core.PySurfaceEnv instance."""
        assert env._rust_env is not None

    def test_rust_env_is_py_surface_env(self, env):
        assert type(env._rust_env).__name__ == "PySurfaceEnv"

    def test_possible_agents_match_config(self, env):
        assert env.possible_agents == ["agent_0", "agent_1"]

    def test_reset_returns_obs_for_all_agents(self, env):
        obs, infos = env.reset(seed=0)
        assert set(obs.keys()) == {"agent_0", "agent_1"}
        assert set(infos.keys()) == {"agent_0", "agent_1"}

    def test_reset_obs_has_correct_keys(self, env):
        obs, _ = env.reset(seed=0)
        for agent_obs in obs.values():
            assert set(agent_obs.keys()) == {"position", "target", "reached"}

    def test_reset_obs_dtypes(self, env):
        obs, _ = env.reset(seed=0)
        for agent_obs in obs.values():
            assert "steps_remaining" not in agent_obs
            assert agent_obs["position"].dtype == np.float32
            assert agent_obs["target"].dtype == np.float32

    def test_reset_steps_remaining_is_not_exposed(self, env):
        obs, _ = env.reset(seed=0)
        for agent_obs in obs.values():
            assert "steps_remaining" not in agent_obs

    def test_step_returns_expected_keys(self, env):
        env.reset(seed=0)
        obs, rewards, terminations, truncations, infos = env.step({"agent_0": 0, "agent_1": 0})
        assert set(obs.keys()) == {"agent_0", "agent_1"}
        assert set(rewards.keys()) == {"agent_0", "agent_1"}

    def test_step_rewards_are_float(self, env):
        env.reset(seed=0)
        _, rewards, _, _, _ = env.step({"agent_0": 0, "agent_1": 0})
        for r in rewards.values():
            assert isinstance(r, float)

    def test_agents_list_clears_on_done(self, env):
        env.reset(seed=0)
        for _ in range(SURFACE_CONFIG["max_steps"] + 1):
            _, _, terminations, _, _ = env.step({"agent_0": 0, "agent_1": 0})
            if all(terminations.values()):
                break
        assert env.agents == []

    def test_episode_terminates_within_max_steps(self, env):
        env.reset(seed=0)
        done = False
        for _ in range(SURFACE_CONFIG["max_steps"] + 1):
            _, _, terminations, _, _ = env.step({"agent_0": 0, "agent_1": 0})
            if all(terminations.values()):
                done = True
                break
        assert done

    def test_direct_rust_env_from_surface_coords(self):
        """Call PySurfaceEnv directly with explicit surface coordinates."""
        rust_env = theseo_core.PySurfaceEnv.from_surface_coords(
            LINE_SURFACE,
            agent_count=1,
            max_steps=30,
        )
        obs = rust_env.reset(seed=42)
        assert obs.total_agents == 1
        assert obs.steps_remaining == 30
        assert len(obs.agents) == 1

    def test_direct_rust_env_agent_state_attributes(self):
        rust_env = theseo_core.PySurfaceEnv.from_surface_coords(
            LINE_SURFACE,
            agent_count=1,
            max_steps=30,
        )
        obs = rust_env.reset(seed=0)
        agent = obs.agents[0]
        assert isinstance(agent.current, tuple) and len(agent.current) == 3
        assert isinstance(agent.target, tuple) and len(agent.target) == 3
        assert isinstance(agent.reached, bool)

    def test_direct_rust_env_step_decrements_steps_remaining(self):
        rust_env = theseo_core.PySurfaceEnv.from_surface_coords(
            LINE_SURFACE,
            agent_count=1,
            max_steps=10,
        )
        rust_env.reset(seed=0)
        result = rust_env.step(0)
        assert result.observation.steps_remaining == 9
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_direct_rust_env_done_at_max_steps(self):
        rust_env = theseo_core.PySurfaceEnv.from_surface_coords(
            LINE_SURFACE,
            agent_count=1,
            max_steps=3,
        )
        rust_env.reset(seed=0)
        rust_env.step(0)
        rust_env.step(0)
        result = rust_env.step(0)  # step 3 of 3
        assert result.done is True

    def test_direct_rust_env_agent_eventually_reaches_goal(self):
        """Single agent on a short line should reach its target before max_steps."""
        rust_env = theseo_core.PySurfaceEnv.from_surface_coords(
            LINE_SURFACE,
            agent_count=1,
            max_steps=100,
        )
        rust_env.reset(seed=0)
        for _ in range(100):
            result = rust_env.step(0)
            if result.observation.reached_agents == 1:
                break
        assert result.observation.reached_agents == 1

    def test_direct_rust_env_deterministic_reset(self):
        """Same seed on the same env instance must give identical assignments."""
        rust_env = theseo_core.PySurfaceEnv.from_surface_coords(
            LINE_SURFACE,
            agent_count=2,
            max_steps=50,
        )
        obs1 = rust_env.reset(seed=42)
        snap1 = [(a.current, a.target) for a in obs1.agents]
        obs2 = rust_env.reset(seed=42)
        snap2 = [(a.current, a.target) for a in obs2.agents]
        assert snap1 == snap2
