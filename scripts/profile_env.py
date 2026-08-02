"""
Environment profiling script.

Measures throughput and identifies bottlenecks at each layer:
  1. Raw Rust step (PyO3 boundary only)
  2. Full VoxelEnv.step (Rust + obs conversion + array allocation)
  3. Full training iteration timers from RLlib result dict
  4. cProfile of one training iteration to locate Python hotspots

Usage (from repo root, with theseo_core wheel installed):
    .venv/Scripts/python.exe scripts/profile_env.py
    .venv/Scripts/python.exe scripts/profile_env.py --rllib   # also runs RLlib iteration
    .venv/Scripts/python.exe scripts/profile_env.py --pstats  # dumps cProfile to profile.out
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
import io
from pathlib import Path

N_STEPS = 5_000   # steps for micro-benchmarks
N_RESET = 200     # resets for reset benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(label: str, elapsed_s: float, n: int) -> None:
    rate = n / elapsed_s
    us   = elapsed_s / n * 1e6
    print(f"  {label:<40s}  {rate:>10,.0f} steps/s   {us:>8.2f} µs/step")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# 1. Raw PyO3 boundary
# ---------------------------------------------------------------------------

def bench_raw_rust(env_cfg: dict) -> None:
    """Benchmark raw Rust environment stepping through the PyO3 boundary.

    Parameters
    ----------
    env_cfg : dict
        Environment configuration dictionary used to initialize the benchmark.

    Returns
    -------
    None
        This function prints throughput metrics to stdout.
    """
    _section("1. Raw Rust step (PyO3 boundary)")
    import theseo_core

    rust_env = theseo_core.PyVoxelEnv(
        max_steps=env_cfg["max_steps"],
    )
    rust_env.reset(42)

    # Warm up
    for _ in range(100):
        result = rust_env.step(0)
        if result.done:
            rust_env.reset(42)

    t0 = time.perf_counter()
    for i in range(N_STEPS):
        result = rust_env.step(i % 26)
        if result.done:
            rust_env.reset(42)
    elapsed = time.perf_counter() - t0
    _bar("rust_env.step()", elapsed, N_STEPS)

    # Attribute access cost
    rust_env.reset(42)
    result = rust_env.step(0)
    obs = result.observation
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        _ = obs.steps_remaining
        _ = obs.filled
    elapsed = time.perf_counter() - t0
    _bar("2x attribute access (steps_remaining, filled)", elapsed, N_STEPS)


# ---------------------------------------------------------------------------
# 2. Full Python env step
# ---------------------------------------------------------------------------

def bench_voxel_env(env_cfg: dict) -> None:
    """Benchmark full Python ``VoxelEnv.step`` execution.

    Parameters
    ----------
    env_cfg : dict
        Environment configuration dictionary used to initialize the benchmark.

    Returns
    -------
    None
        This function prints throughput metrics to stdout.
    """
    _section("2. Full VoxelEnv.step (Rust + obs conversion)")
    from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

    env = VoxelEnv(env_cfg)
    env.reset(seed=42)

    # Warm up
    for _ in range(100):
        _, _, done, _, _ = env.step(env.action_space.sample())
        if done:
            env.reset(seed=42)

    t0 = time.perf_counter()
    for i in range(N_STEPS):
        _, _, done, _, _ = env.step(i % 26)
        if done:
            env.reset(seed=42)
    elapsed = time.perf_counter() - t0
    _bar("VoxelEnv.step() scalar obs", elapsed, N_STEPS)

    # Box obs mode
    box_cfg = {**env_cfg, "obs_mode": "box", "box_radius": 2}
    env_box = VoxelEnv(box_cfg)
    env_box.reset(seed=42)
    t0 = time.perf_counter()
    for i in range(N_STEPS):
        _, _, done, _, _ = env_box.step(i % 26)
        if done:
            env_box.reset(seed=42)
    elapsed = time.perf_counter() - t0
    _bar("VoxelEnv.step() box obs (r=2)", elapsed, N_STEPS)

    # Reset cost
    t0 = time.perf_counter()
    for i in range(N_RESET):
        env.reset(seed=i)
    elapsed = time.perf_counter() - t0
    _bar("VoxelEnv.reset()", elapsed, N_RESET)


# ---------------------------------------------------------------------------
# 3. _obs_to_numpy in isolation (allocation cost)
# ---------------------------------------------------------------------------

def bench_obs_conversion(env_cfg: dict) -> None:
    """Benchmark observation conversion and allocation overhead in isolation.

    Parameters
    ----------
    env_cfg : dict
        Environment configuration dictionary used to initialize the benchmark.

    Returns
    -------
    None
        This function prints throughput metrics to stdout.
    """
    _section("3. _obs_to_numpy allocation cost")
    import theseo_core
    from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

    rust_env = theseo_core.PyVoxelEnv(
        max_steps=env_cfg["max_steps"],
    )
    rust_obs = rust_env.reset(42)

    env = VoxelEnv(env_cfg)
    env._rust_env = rust_env

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        env._obs_to_numpy(rust_obs)
    elapsed = time.perf_counter() - t0
    _bar("_obs_to_numpy() scalar", elapsed, N_STEPS)

    # Box mode
    box_cfg = {**env_cfg, "obs_mode": "box", "box_radius": 2}
    env_box = VoxelEnv(box_cfg)
    env_box._rust_env = rust_env
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        env_box._obs_to_numpy(rust_obs)
    elapsed = time.perf_counter() - t0
    _bar("_obs_to_numpy() box (r=2)", elapsed, N_STEPS)


# ---------------------------------------------------------------------------
# 4. cProfile of one full VoxelEnv episode
# ---------------------------------------------------------------------------

def profile_episode(env_cfg: dict, output_file: str | None = None) -> None:
    """Profile repeated environment episodes with ``cProfile``.

    Parameters
    ----------
    env_cfg : dict
        Environment configuration dictionary used to initialize the profiler.
    output_file : str | None
        Optional output path for serialized profiling data.

    Returns
    -------
    None
        This function prints the top cumulative profile entries and may write a
        profile artifact when ``output_file`` is provided.
    """
    _section("4. cProfile — one full VoxelEnv episode (200 steps)")
    from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

    env = VoxelEnv(env_cfg)

    def _run_episode():
        env.reset(seed=42)
        done = False
        i = 0
        while not done:
            _, _, done, _, _ = env.step(i % 26)
            i += 1

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):   # 50 episodes for statistical weight
        _run_episode()
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(25)
    print(stream.getvalue())

    if output_file:
        pr.dump_stats(output_file)
        print(f"  [saved cProfile data to {output_file}]")
        print(f"  [view with: snakeviz {output_file}]")


# ---------------------------------------------------------------------------
# 5. RLlib iteration timers (optional — requires Ray)
# ---------------------------------------------------------------------------

def bench_rllib_timers(env_cfg: dict) -> None:
    """Run one RLlib iteration and print trainer timing metrics.

    Parameters
    ----------
    env_cfg : dict
        Environment configuration dictionary used to initialize the trainer.

    Returns
    -------
    None
        This function prints timing information to stdout.
    """
    _section("5. RLlib iteration timer breakdown")
    import ray
    from theseo_anysearch.models import (
        Settings, EnvConfig, TrainingConfig, AnyscaleConfig, ModelConfig
    )
    from theseo_anysearch.rllib.algorithms.models import PPOConfig
    from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer

    settings = Settings(
        env=EnvConfig(**{k: v for k, v in env_cfg.items()
                         if k in EnvConfig.model_fields}),
        training=TrainingConfig(
            algorithm="ppo",
            iterations=3,
            checkpoint_interval=999,
            output_dir=Path("runtime/profile"),
            video_every=999,
        ),
        anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
        algorithm_config=PPOConfig(train_batch_size=512, num_sgd_iter=2),
        model_config=ModelConfig(hidden_sizes=[64, 64]),
    )

    trainer = PPOTrainer(settings)
    algo = trainer._build_algorithm()

    print(f"\n  {'Metric':<35s}  {'Value':>12s}")
    print(f"  {'─'*35}  {'─'*12}")

    for i in range(3):
        t0 = time.perf_counter()
        result = algo.train()
        wall = time.perf_counter() - t0

        timers = result.get("timers", {})
        print(f"\n  Iteration {i+1}  (wall: {wall:.2f}s)")
        for key in ("sample_time_ms", "learn_time_ms", "update_time_ms",
                    "training_iteration_s"):
            if key in timers:
                print(f"    {key:<35s}  {timers[key]:>10.1f}")

        throughput = result.get("num_env_steps_sampled_this_iter", 0)
        if throughput:
            print(f"    {'steps_sampled_this_iter':<35s}  {throughput:>10,d}")
        ep_mean = (result.get("env_runners", {}).get("episode_return_mean")
                   or result.get("episode_reward_mean", 0))
        print(f"    {'episode_reward_mean':<35s}  {ep_mean:>10.3f}")

    ray.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the selected profiling routines.

    Returns
    -------
    None
        This function orchestrates the profiling workflow and prints results.
    """
    parser = argparse.ArgumentParser(description="Profile anysearch environments")
    parser.add_argument("--rllib",  action="store_true", help="Also run RLlib iteration timers (starts Ray)")
    parser.add_argument("--pstats", action="store_true", help="Dump cProfile output to profile.out")
    args = parser.parse_args()

    env_cfg = {
        "max_steps": 200,
        "seed": 42,
        "agent_count": 4,
        "obs_mode": "scalar",
        "box_radius": 2,
        "ray_max_len": 16,
        "grid_size": 32,
        "step_cost": -0.01,
        "goal_reward": 1.0,
        "distance_shaping": 0.0,
        "collision_cost": 0.0,
        "trail_mode": False,
        "geometry_boxes": None,
        "waypoints_file": None,
    }

    print("\n" + "═" * 60)
    print("  anysearch environment profiler")
    print("═" * 60)

    bench_raw_rust(env_cfg)
    bench_voxel_env(env_cfg)
    bench_obs_conversion(env_cfg)
    profile_episode(env_cfg, output_file="profile.out" if args.pstats else None)

    if args.rllib:
        bench_rllib_timers(env_cfg)

    print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    main()
