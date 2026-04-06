# Usage

Example geometries and experiment configurations for `anysearch`.

```
usage/
├── geometries/          # Sample ASCII STL files
│   ├── README.md        # Geometry descriptions and scale guidance
│   ├── cube.stl
│   ├── stepped_terrain.stl
│   ├── corridor_l.stl
│   ├── pipe_junction.stl
│   └── ramp_spiral.stl
└── experiments/
    ├── train/           # Fixed-config training runs
    │   ├── ppo_baseline.yaml        # PPO on stepped terrain
    │   ├── sac_baseline.yaml        # SAC on pipe junction
    │   ├── ppo_corridor.yaml        # PPO on L-corridor routing
    │   └── ppo_spiral.yaml          # PPO on 3-D spiral ramp
    └── tune/            # Hyperparameter search + sweeps
        ├── ppo_asha.yaml            # PPO × ASHA (lr + batch size)
        ├── ppo_pbt.yaml             # PPO × PBT (evolutionary)
        ├── sac_asha.yaml            # SAC × ASHA (off-policy search)
        └── ppo_sweep_geometries.yaml # Sweep: same PPO config across all 4 geometries
```

## Quick start

```bash
# single training run
anysearch experiment run --config usage/experiments/train/ppo_baseline.yaml

# hyperparameter search
anysearch experiment run --config usage/experiments/tune/ppo_asha.yaml

# geometry sweep
anysearch experiment run --config usage/experiments/tune/ppo_sweep_geometries.yaml

# resume an interrupted run
anysearch experiment resume --run-id <run_id>

# inspect results
anysearch experiment inspect --run-id <run_id>
```

All configs expect MLflow at `http://localhost:5000`.
Start a local server with:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Or remove the `mlflow.tracking_uri` key to fall back to a local `./mlruns` directory.
