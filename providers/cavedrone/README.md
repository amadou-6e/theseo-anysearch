# CaveDrone voxel-world provider

This optional wheel registers `cavedrone` under `anysearch worlds`. It wraps
the repository's source-faithful CaveDroneSim exporter; it does not bundle,
download, or compile upstream source during pip installation.

Prerequisites: `theseo-anysearch` 0.1.x, Git, and a C++23 compiler (`g++` by
default). Explicitly obtain the MIT-licensed upstream
[`makarov-mm/cave-drone`](https://github.com/makarov-mm/cave-drone) checkout at
`ef7852198249390806d8c0cd42e576e02c73c19f`, inspect its rights, then set
`ANYSEARCH_CAVEDRONE_SOURCE` to its directory. The provider verifies that
revision, a clean tracked tree, the license notice, and source-file SHA-256
values before every generation. Set `ANYSEARCH_CAVEDRONE_CXX` only when `g++`
is not the intended compiler.

```powershell
python -m pip install .\providers\cavedrone
$env:ANYSEARCH_CAVEDRONE_SOURCE = 'C:\path\to\cave-drone'
anysearch worlds list
anysearch worlds cavedrone --seed 42 --output worlds/cave-42
anysearch worlds add worlds/cave-42 --config experiments/train.yaml
```

The native output is a 192 x 56 x 192 full-occupancy grid with 0.5 m voxels.
`--partition` may be `train` (default), `validation`, or `test`, and
`--body-radius-m` defaults to 0.25. There is no native voxel-resolution knob.
The core verifies every derived six-axis swept-sphere route against complete
occupied voxel cubes, then writes six PNG views and `verification.json`.
Rejected task strata are reported. A seed changes a layout within one
generator family, not the topology family. Neither full truth nor the derived
fixed-goal routes are native sensor evidence, continuous flight validation,
or the upstream exploration-and-return task.

The initial `worlds add` path selects one **train** world and one deterministic
task per YAML. It does not create a runnable multi-world train/test study.
See `docs/world_providers.md` for the shared-study-ID audit example.
