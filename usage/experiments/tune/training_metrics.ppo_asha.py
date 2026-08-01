"""Training-efficiency metrics used by ``ppo_asha.yaml``."""


def compute_metrics(context):
    mean_length = context.standard_metrics["episode_len_mean"]
    mean_reward = context.standard_metrics["episode_reward_mean"]
    episodes_total = context.standard_metrics["episodes_total"]
    environment_steps = context.environment_steps_total

    return {
        "reward_per_episode_step": (
            mean_reward / mean_length if mean_length > 0 else 0.0
        ),
        "episodes_per_1000_env_steps": (
            1000.0 * episodes_total / environment_steps
            if environment_steps > 0
            else 0.0
        ),
    }
