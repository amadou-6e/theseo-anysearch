"""Runtime helpers shared by RLlib algorithm adapters and trainers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
def _log_trainer_stage(message: str) -> None:
    """Print a timestamped trainer progress message for foreground debugging."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[trainer] {ts} {message}", flush=True)


def _append_trainer_stage_log(output_dir: Path, message: str) -> None:
    """Append a timestamped trainer stage marker to the run-local debug log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = output_dir.joinpath("debug_stage.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[trainer] {ts} {message}\n")


def _resolve_pool_dir(geometry_pool: dict | None) -> dict | None:
    """Return geometry_pool with pool_dir resolved to an absolute path.

    The CLI resolves relative pool_dir values before calling any trainer, so
    this is typically a no-op.  It exists as a safety net for callers that
    build trainers directly (e.g. tests) with relative paths.
    """
    if not geometry_pool or not geometry_pool.get("pool_dir"):
        return geometry_pool
    pool_dir = Path(str(geometry_pool["pool_dir"]))
    if not pool_dir.is_absolute():
        pool_dir = Path(Path.cwd(), pool_dir)
    return {**geometry_pool, "pool_dir": str(pool_dir.resolve())}


def _patch_rllib_replay_buffer_type_check() -> None:
    """
    Work around a Ray bug in Algorithm._create_local_replay_buffer_if_necessary.

    After AlgorithmConfig.validate() resolves replay_buffer_config["type"] from a
    string to the actual class, the legacy check
        if "EpisodeReplayBuffer" in config["replay_buffer_config"]["type"]
    fails with TypeError because "in" on a class object is not supported.

    This patch converts the resolved class back to its name string before the
    check so that the string membership test works correctly.
    """
    try:
        import ray.rllib.algorithms.algorithm as _alg_mod
        _orig = _alg_mod.Algorithm._create_local_replay_buffer_if_necessary

        def _safe(self, config):  # type: ignore[override]
            rb = config.get("replay_buffer_config")
            if rb and not isinstance(rb.get("type"), str):
                config = dict(config, replay_buffer_config=dict(rb, type=rb["type"].__name__))
            return _orig(self, config)

        _alg_mod.Algorithm._create_local_replay_buffer_if_necessary = _safe
    except Exception:
        pass  # Ray not installed or API changed — skip silently


_patch_rllib_replay_buffer_type_check()


def _detect_num_gpus(require_gpu: bool = False, *, num_gpus: float | None = None) -> float:
    """Return GPU count to allocate per RLlib algorithm.

    If *num_gpus* is provided it is returned directly (allows fractional values
    for concurrent Tune trials sharing one GPU, e.g. 0.5).
    When *require_gpu* is True, raises AssertionError if no CUDA device is found.
    Install hint: pip install .[torch-gpu] --index-url https://download.pytorch.org/whl/cu124
    """
    if num_gpus is not None:
        return num_gpus
    import torch

    n = torch.cuda.device_count()
    if require_gpu:
        assert n > 0, (
            f"require_gpu=True but torch.cuda.device_count()={n}. "
            "Install CUDA-enabled torch: pip install .[torch-gpu] "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
    return n
