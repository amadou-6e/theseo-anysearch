"""Compatibility imports for the relocated PPO algorithm adapter."""

# Import the remaining built-ins to preserve the historical side effect of
# importing trainer.ppo: every bundled adapter becomes available in
# Trainer._registry. Canonical algorithm-package imports remain lazy.
from theseo_anysearch.rllib.algorithms.ddpg import DDPGTrainer as _DDPGTrainer
from theseo_anysearch.rllib.algorithms.dqn import DQNTrainer as _DQNTrainer
from theseo_anysearch.rllib.algorithms.multi_voxel_ppo import (
    MultiAgentVoxelPPOTrainer as _MultiAgentVoxelPPOTrainer,
)
from theseo_anysearch.rllib.algorithms.rainbow import RainbowTrainer as _RainbowTrainer
from theseo_anysearch.rllib.algorithms.sac import SACTrainer as _SACTrainer
from theseo_anysearch.rllib.algorithms.td3 import TD3Trainer as _TD3Trainer
from theseo_anysearch.rllib.algorithms.ppo import (
    PPOTrainer,
    VoxelEnvPathHelper,
    _build_rllib_ppo,
    _configure_rllib_env_runners,
    _ensure_ray_runtime,
    _set_rllib_storage_path,
)

__all__ = ["PPOTrainer", "VoxelEnvPathHelper"]
