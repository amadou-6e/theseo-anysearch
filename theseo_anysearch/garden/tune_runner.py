"""Ray Tune hyperparameter search for garden encoder pre-training."""
from __future__ import annotations

from pathlib import Path
from typing import Any


_VALID_TUNE_FNS = {"choice", "grid_search", "loguniform", "uniform", "randint", "randn"}
_CHOICE_FNS = {"choice", "grid_search"}


def _parse_garden_search_space(raw: dict[str, Any], tune_module: Any) -> dict[str, Any]:
    """Convert garden YAML search_space to ray.tune samplers.

    YAML format — top-level keys are GardenConfig section names
    (``encoder``, ``training``, ``augmentation``), values are param dicts::

        encoder:
          latent_dim:    {choice: [32, 64, 128]}
          conv_channels: {choice: [[16, 32], [32, 64], [32, 64, 64]]}
        training:
          lr:         {loguniform: [1.0e-4, 1.0e-2]}
          batch_size: {choice: [64, 128, 256]}

    Returns a flat dict ``{"encoder.latent_dim": tune.choice([...]), ...}``
    which is passed as the ``param_space`` to ``ray.tune.Tuner``.
    """
    result: dict[str, Any] = {}
    for section, params in raw.items():
        if not isinstance(params, dict):
            continue
        for param, spec in params.items():
            key = f"{section}.{param}"
            if not isinstance(spec, dict) or len(spec) != 1:
                raise ValueError(f"search_space.{key}: expected {{fn: [args]}}, got {spec!r}")
            fn_name, args = next(iter(spec.items()))
            if fn_name not in _VALID_TUNE_FNS:
                raise ValueError(f"search_space.{key}: unknown tune fn '{fn_name}'")
            fn = getattr(tune_module, fn_name)
            result[key] = fn(args) if fn_name in _CHOICE_FNS else fn(*args)
    return result


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set d["a"]["b"]["c"] = value for dotted_key="a.b.c"."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _resolve_paths_in_dict(d: Any, base: Path) -> Any:
    """Recursively resolve relative path strings in a dict to absolute.

    Tries CWD first (paths relative to project root), then *base* (config file parent).
    """
    if isinstance(d, dict):
        return {k: _resolve_paths_in_dict(v, base) for k, v in d.items()}
    if isinstance(d, list):
        return [_resolve_paths_in_dict(v, base) for v in d]
    if isinstance(d, str):
        p = Path(d)
        if not p.is_absolute():
            cwd = Path.cwd()
            for root in (cwd, base):
                candidate = root / p
                if candidate.exists():
                    return str(candidate.resolve())
    return d


def _garden_trainable(config: dict[str, Any], *, garden_dict: dict[str, Any]) -> None:
    """Ray Tune trainable: train one garden encoder trial and report val_loss each epoch."""
    import copy
    from ray import tune as _tune
    from theseo_anysearch.garden.collect import collect_dataset
    from theseo_anysearch.garden.data_config import GardenConfig, load_data_config
    from theseo_anysearch.garden.dataset import make_train_val_datasets
    from theseo_anysearch.garden.encoder import build_model
    from theseo_anysearch.garden.trainer import train as run_train

    # Apply sampled hyperparams onto the base config dict
    merged = copy.deepcopy(garden_dict)
    for dotted_key, value in config.items():
        _deep_set(merged, dotted_key, value)

    # Remove tune block so GardenConfig validates cleanly
    merged.pop("tune", None)

    cfg = GardenConfig.model_validate(merged)

    if isinstance(cfg.data, Path):
        data_cfg = load_data_config(cfg.data)
    else:
        data_cfg = cfg.data

    grids = collect_dataset(data_cfg, box_radius=cfg.encoder.box_radius)
    train_ds, val_ds = make_train_val_datasets(grids, cfg.augmentation, data_cfg.split)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg)

    def epoch_cb(epoch: int, train_loss: float, val_loss: float) -> None:
        _tune.report({"val_loss": val_loss, "train_loss": train_loss, "epoch": epoch})

    run_train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg.training,
        approach=cfg.approach,
        device=device,
        epoch_cb=epoch_cb,
    )


class GardenTuneRunner:
    """Run Ray Tune sweeps for garden encoder pretraining.

    Parameters
    ----------
    config_path : Path
        Path to the garden config YAML file.
    tag : str | None
        Optional run tag appended to the sweep name.
    """

    def __init__(self, config_path: Path, tag: str | None = None) -> None:
        from theseo_anysearch.garden.data_config import load_garden_config
        self._cfg = load_garden_config(config_path)
        self._config_path = config_path
        self._tag = tag

        if self._cfg.tune is None:
            raise ValueError(f"No 'tune:' block in {config_path}. Add tune config to enable garden tune.")

    def run(self, output_dir: Path) -> dict[str, Any]:
        import ray
        from ray import tune as ray_tune
        from ray.tune.schedulers import AsyncHyperBandScheduler

        tc = self._cfg.tune

        # Serialise config and resolve ALL paths to absolute so Ray workers
        # (which run with a different CWD) can find them.
        cfg_dict = self._cfg.model_dump(mode="json")
        cfg_dict = _resolve_paths_in_dict(cfg_dict, self._config_path.parent)

        search_space = _parse_garden_search_space(tc.search_space, ray_tune)

        scheduler = AsyncHyperBandScheduler(
            max_t=self._cfg.training.epochs,
            grace_period=tc.grace_period,
            reduction_factor=tc.reduction_factor,
        )

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        import torch
        gpu_per_trial = (1.0 / tc.max_concurrent) if torch.cuda.is_available() else 0.0

        trainable = ray_tune.with_resources(
            ray_tune.with_parameters(_garden_trainable, garden_dict=cfg_dict),
            resources={"cpu": 1, "gpu": gpu_per_trial},
        )

        run_name = f"{self._cfg.name}_tune" + (f"_{self._tag}" if self._tag else "")

        # Short trial dir names to stay under Windows MAX_PATH (260 chars)
        trial_counter = [0]
        def _short_dirname(trial):
            trial_counter[0] += 1
            return f"t{trial_counter[0]:03d}"

        tuner = ray_tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=ray_tune.TuneConfig(
                metric="val_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=tc.num_samples,
                max_concurrent_trials=tc.max_concurrent,
                trial_dirname_creator=_short_dirname,
            ),
            run_config=ray_tune.RunConfig(
                name=run_name,
                storage_path=str(output_dir.resolve()),
            ),
        )

        results = tuner.fit()
        best = results.get_best_result(metric="val_loss", mode="min")
        return {
            "best_config": best.config,
            "best_val_loss": best.metrics.get("val_loss"),
            "run_name": run_name,
            "num_trials": len(results),
        }

    @property
    def _cfg_path(self) -> Path:
        return self._config_path
