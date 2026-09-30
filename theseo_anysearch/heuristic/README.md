# Heuristic package

The package separates reusable planning mechanics from concrete search
strategies:

```text
heuristic/
├── models.py              # Immutable plan and replay results
├── base.py                # Voxel graph construction and replay mechanics
└── voxel/
    ├── astar/
    │   ├── standard.py
    │   ├── weighted.py
    │   └── replanning.py
    ├── dijkstra.py
    └── factory.py         # Stable YAML-name dispatch
```

Public users should import strategies and result models from
`theseo_anysearch.heuristic`. Internal code may import the canonical module
that owns an implementation.

## Adding a voxel heuristic

1. Add one strategy module under `voxel/`.
2. Inherit `BaseVoxelHeuristic` and implement `_find_path`, or inherit an
   existing strategy when only replay behavior changes.
3. Register the YAML-facing name in `voxel/factory.py`.
4. Export the strategy from `voxel/__init__.py` and the package `__init__.py`.
5. Add focused strategy and factory-dispatch tests under
   `tests/test_heuristic/`.

YAML names are part of the experiment configuration contract. Renaming one
requires an explicit compatibility migration.
