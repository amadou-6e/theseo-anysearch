"""Concrete RLlib algorithm configuration models and lookup helpers."""

from __future__ import annotations

from pydantic import ConfigDict

from theseo_anysearch.settings import AlgorithmConfig


class PPOConfig(AlgorithmConfig):
    """PPO-specific algorithm configuration."""
    model_config = ConfigDict(extra="forbid")

    clip_param: float = 0.2
    minibatch_size: int | None = None
    num_sgd_iter: int = 10
    lambda_: float = 0.95
    kl_coeff: float = 0.2
    grad_clip: float = 40.0     # max global gradient norm; prevents NaN from high LRs


class SACConfig(AlgorithmConfig):
    """SAC-specific algorithm configuration."""
    model_config = ConfigDict(extra="forbid")

    tau: float = 0.005
    target_entropy: str = "auto"
    n_step: int = 1
    warmup_steps: int = 0


class DQNConfig(AlgorithmConfig):
    """DQN-specific algorithm configuration.

    Note: train_batch_size is the replay buffer sample batch size per gradient
    update, not the on-policy rollout length (as in PPO).
    """
    model_config = ConfigDict(extra="forbid")

    n_step: int = 1
    num_atoms: int = 1          # 1 = vanilla DQN; >1 = distributional
    dueling: bool = True
    double_q: bool = True
    noisy: bool = False
    replay_buffer_capacity: int = 50_000
    warmup_steps: int = 0       # num_steps_sampled_before_learning_starts


class APPOConfig(AlgorithmConfig):
    """APPO-specific asynchronous actor-learner configuration."""

    model_config = ConfigDict(extra="forbid")

    rollout_fragment_length: int = 32
    vtrace: bool = True
    use_gae: bool = True
    lambda_: float = 1.0
    clip_param: float = 0.4
    use_kl_loss: bool = False
    kl_coeff: float = 1.0
    kl_target: float = 0.01
    target_network_update_freq: int = 2
    circular_buffer_num_batches: int = 8
    circular_buffer_iterations_per_batch: int = 2


class RainbowConfig(DQNConfig):
    """Rainbow DQN: DQN + distributional Q + noisy nets + PER."""
    model_config = ConfigDict(extra="forbid")

    n_step: int = 3
    num_atoms: int = 51
    noisy: bool = True
    v_min: float = -10.0
    v_max: float = 10.0
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta: float = 0.4


class TD3Config(AlgorithmConfig):
    """TD3-specific algorithm configuration.

    Requires a continuous (Box) action space environment.
    The current VoxelEnv and SurfaceEnv use Discrete action spaces.
    """
    model_config = ConfigDict(extra="forbid")

    twin_q: bool = True
    policy_delay: int = 2
    smooth_target_policy: bool = True
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    tau: float = 0.005


class DDPGConfig(AlgorithmConfig):
    """DDPG-specific algorithm configuration.

    Requires a continuous (Box) action space environment.
    The current VoxelEnv and SurfaceEnv use Discrete action spaces.
    """
    model_config = ConfigDict(extra="forbid")

    tau: float = 0.002
    l2_reg: float = 1e-6


ALGORITHM_CONFIGS: dict[str, type[AlgorithmConfig]] = {
    "appo": APPOConfig,
    "ppo": PPOConfig,
    "sac": SACConfig,
    "dqn": DQNConfig,
    "rainbow": RainbowConfig,
    "td3": TD3Config,
    "ddpg": DDPGConfig,
    "multi_agent_voxel_ppo": PPOConfig,
}


def get_algorithm_config_class(name: str) -> type[AlgorithmConfig]:
    """Return the concrete algorithm config class for a registered algorithm name.

    Parameters
    ----------
    name : str
        Registered algorithm name.

    Returns
    -------
    type[AlgorithmConfig]
        Matching config class, or ``AlgorithmConfig`` when the name is unknown.
    """
    return ALGORITHM_CONFIGS.get(name.lower(), AlgorithmConfig)
