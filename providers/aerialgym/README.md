# Aerial Gym voxel provider

Install the core extra: `python -m pip install "theseo-anysearch[aerialgym]"`.
Until release, build both wheels and install the extra with pip `--find-links`
pointing to the local wheel directory. Installing pip packages never downloads
assets or generates worlds.

```powershell
anysearch worlds list
anysearch worlds aerialgym --seed 42 --meters-per-voxel 0.25 --output worlds/aerial-42-025
anysearch worlds aerialgym --seed 42 --meters-per-voxel 0.10 --offline true --output worlds/aerial-42-010
anysearch worlds aerialgym --seed 42 --layout altitude --output worlds/aerial-altitude-42
anysearch worlds add worlds/aerial-42-025 --config experiments/train.yaml
```

Generation downloads ten allowlisted hash-pinned files, not the simulator.
Cache location: `ANYSEARCH_AERIALGYM_CACHE` or `~/.cache/anysearch/aerialgym`.
Every reuse verifies source hashes. Offline mode refuses an empty cache.
Resolution is 0.10 to 0.50 meters and must divide both 10 and 6 meters.
Bodies default to 0.25 meter spheres. Unsupported URDF bodies fail closed.

Worlds use conservative rotated-box occupancy, fixed bounds and seeded derived
static recipes, not captured native Aerial Gym scenes, sensors or controllers.
Task acceptance is checked independently at each resolution. The BSD notice
is preserved in each bundle. Different resolutions share a root geometry and
must not be split across train/test roles. Current CLI attaches one world and
one selected task per training YAML; generation does not start training.

Governing merged spec: `amadou-6e/specs@f5e701b23c9c52a01f48ca7b76ed7be89a62268a`.
