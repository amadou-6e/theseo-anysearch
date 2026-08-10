"""Runtime extensions for reducing modern DQN driver overhead."""

from __future__ import annotations

from typing import Any

from ray.rllib.algorithms.dqn import DQN as RllibDQN
from ray.rllib.algorithms.dqn.dqn import (
    DQNConfig as RllibDQNConfig,
    TD_ERROR_KEY,
)
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.core import ALL_MODULES
from ray.rllib.utils.annotations import override


class AnySearchDQNTorchLearner(DQNTorchLearner):
    """Suppress per-sample TD-error transport when replay is uniform."""

    @override(DQNTorchLearner)
    def compute_loss_for_module(self, **kwargs: Any) -> Any:
        """Compute DQN loss and retain TD errors only when priorities need them."""
        loss = super().compute_loss_for_module(**kwargs)
        if not self.config.learner_config_dict.get("report_td_errors", True):
            module_id = kwargs["module_id"]
            self.metrics.delete(module_id, TD_ERROR_KEY, key_error=False)
        return loss


class AnySearchDQN(RllibDQN):
    """DQN algorithm with configurable EnvRunner weight synchronization."""

    @classmethod
    @override(RllibDQN)
    def get_default_config(cls) -> RllibDQNConfig:
        """Return an RLlib DQN configuration bound to this algorithm class."""
        return RllibDQNConfig(algo_class=cls)

    def setup(self, config: RllibDQNConfig) -> None:
        """Synchronize EnvRunners after a configured number of learner updates."""
        super().setup(config)
        interval = config.learner_config_dict.get("weight_sync_interval", 1)
        original_sync = self.env_runner_group.sync_weights
        original_update = self.learner_group.update
        learner_updates = 0

        def update_and_sync(*args: Any, **kwargs: Any) -> Any:
            nonlocal learner_updates
            results = original_update(*args, **kwargs)
            learner_updates += 1
            if learner_updates % interval == 0:
                modules = set(results[0]) - {ALL_MODULES}
                original_sync(
                    from_worker_or_learner_group=self.learner_group,
                    policies=modules,
                    global_vars=None,
                    inference_only=True,
                )
            return results

        self.learner_group.update = update_and_sync
        self.env_runner_group.sync_weights = lambda *args, **kwargs: None
