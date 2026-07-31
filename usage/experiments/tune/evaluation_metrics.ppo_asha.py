"""Navigation score used by ``ppo_asha.yaml`` for ASHA ranking."""


def compute_metrics(context):
    progress_fractions = []
    for episode in context.episodes:
        initial = episode.initial_goal_distance
        final = episode.final_goal_distance
        if initial is None or final is None or initial <= 0:
            continue
        progress_fractions.append(max(0.0, min(1.0, (initial - final) / initial)))
    progress_mean = (
        sum(progress_fractions) / len(progress_fractions)
        if progress_fractions
        else 0.0
    )
    success_rate = context.standard_metrics["evaluation_success_rate"]
    return {
        "navigation_score": 10.0 * success_rate + progress_mean,
        "goal_progress_fraction_mean": progress_mean,
    }
