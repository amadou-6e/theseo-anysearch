# Verified voxel-world providers

Governing contract:
[`specs@84caf742`](https://github.com/amadou-6e/specs/blob/84caf74220cc82e68e4d314fe8a98239d93f0927/projects/theseo-anysearch/world-provider-cli.md).
This issue implements the provider-neutral CLI and a deterministic fixture;
the CaveDrone wheel and its upstream-source setup belong to #430.

```text
anysearch worlds list
anysearch worlds list --remote
anysearch worlds fixture-boxes --seed 42 --output worlds/fixture-42
anysearch worlds add worlds/fixture-42 --config experiments/train.yaml
```

Generation refuses to overwrite an output directory. It validates the
existing routing sidecars, every task endpoint, every six-axis route segment
against occupied voxel cubes and a solid world boundary, then writes three
orthogonal projections, three task-start slices, and `verification.json`.
`list` is offline and uses a local registry only for discoverability; it
rechecks worlds before displaying `verified`. Set
`ANYSEARCH_WORLDS_REGISTRY` to override its location. `list --remote` reads a
packaged versioned catalog (initially empty) or the JSON file/HTTPS URL in
`ANYSEARCH_WORLDS_CATALOG`; it never installs or runs downloaded code.

Optional pip distributions register an entry point in
`theseo_anysearch.world_providers`. The entry point loads an object or class
with a `ProviderInfo` and `generate(seed=, output=, parameters=)` method
returning `GenerationSummary(rejected_task_strata=...)`.
The generator writes `occupancy.npy`, `source.json`, `conversion.json`,
`world.json`, `split.json`, `task-*.json`, `reference-*.json`, and route JSON
artifacts. It does not write `verification.json` or previews; the core does
that after independent checks. Parameter names/types/bounds and native meters
per voxel are declared in `ProviderInfo`. A broken optional plugin is shown
as unavailable and cannot break the built-in commands.

```toml
[project.entry-points."theseo_anysearch.world_providers"]
my-provider = "my_worlds:Provider"
```

`worlds add` requires a verified world with training rights and a YAML that
does not already select another geometry. It compiles that voxel grid into an
immutable world pack, selects the first task by stable task identity, writes
one checked waypoint file, and records world/split/task identities under the
YAML `worlds` block. The run loader repeats validation and resolves the pack
and waypoints relative to the YAML. The first implementation supports **one
world and one selected task per experiment YAML**; it rejects a second world
instead of suggesting that multiple worlds are sampled at runtime. A
`tune_config` in the same experiment YAML uses the same selected world.
Separate sweep YAMLs and interactive 3D viewing are not included yet.
Configs in one study declare the same `worlds.study_id` and `study_root`.
The loader reads their actual YAML files and requires the same split record;
this first version refuses adding a second world to one study instead of
silently producing incompatible split records. Different YAMLs have distinct
study IDs by default, even in one directory.

The fixture is only an offline plumbing example, not evidence of transfer or
a useful training family. Six-axis swept-sphere replay establishes voxel-cube
collision freedom, not continuous controller feasibility or path optimality.
Other motion/constraint models need an explicitly reviewed independent
checker before they can register verified tasks.
