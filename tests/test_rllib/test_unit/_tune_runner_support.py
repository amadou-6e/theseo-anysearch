"""Support helpers for tune runner unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def make_experiment_config(
    *,
    require_gpu: bool = False,
    num_env_runners: int = 4,
    evaluation_num_env_runners: int = 0,
    max_concurrent: int = 1,
    iterations: int = 10,
    output_dir: str = "/tmp/exp",
):
    """Build a representative experiment config for tune runner tests.

    Parameters
    ----------
    require_gpu : bool, default=False
        Whether the training config should request GPU resources.
    num_env_runners : int, default=4
        Number of Ray env runners for the training config.
    evaluation_num_env_runners : int, default=0
        Number of dedicated Ray evaluation env runners.
    max_concurrent : int, default=1
        Maximum concurrently scheduled tune trials.
    iterations : int, default=10
        Trial training iterations.
    output_dir : str, default="/tmp/exp"
        Experiment output directory.

    Returns
    -------
    ExperimentConfig
        Config object suitable for tune runner construction tests.
    """

    from theseo_anysearch.experiments.models import ExperimentConfig, ExperimentMeta, TuneConfig
    from theseo_anysearch.models import (
        AlgorithmConfig,
        AnyscaleConfig,
        EnvConfig,
        ModelConfig,
        TrainingConfig,
    )

    return ExperimentConfig(
        experiment=ExperimentMeta(name="test-sweep", output_dir=output_dir),
        env=EnvConfig(agent_count=2, max_steps=50),
        training=TrainingConfig(
            algorithm="multi_agent_voxel_ppo",
            iterations=iterations,
            require_gpu=require_gpu,
            num_env_runners=num_env_runners,
            evaluation_num_env_runners=evaluation_num_env_runners,
            output_dir=output_dir,
        ),
        anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
        algorithm_config=AlgorithmConfig(lr=3e-4, gamma=0.99, train_batch_size=512),
        model_cfg=ModelConfig(hidden_sizes=[64]),
        tune_config=TuneConfig(
            scheduler="asha",
            num_samples=4,
            max_concurrent=max_concurrent,
            metric="episode_reward_mean",
            mode="max",
        ),
    )


def make_fake_tuner_fit():
    """Create a fake tuner object whose fit method returns a best result."""

    return MagicMock(
        fit=MagicMock(
            return_value=MagicMock(
                get_best_result=MagicMock(
                    return_value=MagicMock(
                        config={},
                        metrics={"episode_reward_mean": 0.0, "training_iteration": 1},
                    )
                )
            )
        )
    )


def patch_ray_tune(
    fake_with_resources=None,
    fake_with_parameters=None,
    fake_tune_config=None,
):
    """Build the standard patch set used by tune runner construction tests.

    Parameters
    ----------
    fake_with_resources : callable, optional
        Replacement for ``ray.tune.with_resources``.
    fake_with_parameters : callable, optional
        Replacement for ``ray.tune.with_parameters``.
    fake_tune_config : callable, optional
        Replacement for ``ray.tune.TuneConfig``.

    Returns
    -------
    list
        ``unittest.mock.patch`` objects ready to enter with an ``ExitStack``.
    """

    import ray.tune as ray_tune

    return [
        patch("ray.init"),
        patch("ray.shutdown"),
        patch.object(
            ray_tune,
            "with_resources",
            fake_with_resources or MagicMock(return_value=MagicMock()),
        ),
        patch.object(
            ray_tune,
            "with_parameters",
            fake_with_parameters or MagicMock(return_value=MagicMock()),
        ),
        patch.object(
            ray_tune,
            "TuneConfig",
            fake_tune_config or MagicMock(return_value=MagicMock()),
        ),
        patch.object(ray_tune, "RunConfig", MagicMock(return_value=MagicMock())),
        patch.object(ray_tune, "Tuner", MagicMock(return_value=make_fake_tuner_fit())),
        patch("theseo_anysearch.cli.commands.tune._build_scheduler", return_value=MagicMock()),
        patch("theseo_anysearch.cli.commands.tune._build_search_alg", return_value=None),
        patch("theseo_anysearch.cli.commands.tune._parse_search_space", return_value={}),
        patch("theseo_anysearch.experiments.tune_runner._print_sweep_overview"),
    ]


def write_existing_sweep(tmp_path: Path, *, run_tag: str = "latest") -> Path:
    """Create a minimal sweep directory layout for continuation tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory root.
    run_tag : str, default="latest"
        Sweep tag name.

    Returns
    -------
    Path
        Path to the synthetic sweep directory.
    """

    sweep_dir = tmp_path.joinpath("test-sweep", run_tag)
    trial_dir = sweep_dir.joinpath("abc12345")
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.joinpath("ray_runtime.json").write_text("{}", encoding="utf-8")
    sweep_dir.joinpath("tune_runtime.json").write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "ray_storage_root": str(tmp_path.joinpath("ray-store")),
                "active_segment": None,
                "segments": [
                    {
                        "segment_name": "base",
                        "tune_run_name": run_tag,
                        "trial_prefix": "",
                        "num_samples": 4,
                        "status": "COMPLETED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return sweep_dir
