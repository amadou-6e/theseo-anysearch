# Plant01 point-path import diagnostic

Status: technical validation for [#412](https://github.com/amadou-6e/theseo-anysearch/issues/412),
based on integrated [#411](https://github.com/amadou-6e/theseo-anysearch/pull/419)
at `0f6bc476de2fe9ab3d7678a3c1eb52c9cb2ac154`.
The governing roadmap is
[specs@1dc8397](https://github.com/amadou-6e/specs/blob/1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b/projects/theseo-anysearch/routing-geometry-integration-roadmap.md).
This is **not** a published-score comparison or a training run. Source rights
are still unreviewed; no third-party map, scenario, or converted corpus is
committed.

## Source and conversion

| Item | SHA-256 or result |
| --- | --- |
| `plant01.3dmap.zip` | `0aa958fbf25db45d6e9c531ece0c1a2534e06caa4adf34ce301e05fd0d2ad8e3` |
| `plant01.3dscen` | `48eba4bb7e642c882878d0a430ffd70d3ef78c8209e9d530736ad95290c536b1` |
| ZIP member | `plants/plant01.3dmap`, `voxel 195 128 100` |
| Distinct occupied voxels | 79,702 |
| Repeated occupied-coordinate rows | 7,918; idempotent overlaps, not parse errors |
| Conversion identity | `84592832f4380f5d98a31d7bb035cedafc41b14908dfa7a78303392a0eacdc99` |
| Routing-world identity | `6709a283a52f0cea9d4ec027e41c323797ac8253b618432679b2b01a4e344a7d` |
| Compiled `NpySource` world-pack identity | `72b26c768f1c0b43c95ec11b883f54a899bfbaf448ed36811670cd0687ed7891` |
| Routing-bundle identity | `e5d64ca1a4b1247e1fe4dd9431201378d7772f21126a63c3fb0b0204a3a6a4e8` |
| Accepted-query-set identity | `e9c9ab29ce25d87be87882199759d770f01a7c75d88976592e4671417a434312` |

The strict importer validates the member, header, every coordinate, and every
scenario row. It compiles a three-axis `.npy` occupancy source through the
existing world-pack compiler. All 2,000 scenario rows had numeric fields,
in-bounds unoccupied endpoints, and were accepted; zero were rejected. The
three trailing numeric columns are retained as opaque values, not treated as
verified optimal costs.

## Fixed-prefix baseline

The first 16 source queries were selected by line order only. All 16 returned
paths, and a separate replay check verified endpoints, world bounds,
unoccupied voxels, neighboring steps, diagonal corner/edge clearance, and
Euclidean step cost. Search uses NetworkX A* on an implicit 26-neighbor graph
with a voxel-distance heuristic. The first 10 queries were short (1-3 node
expansions); the remaining six used 1,098-7,547 expansions. This prefix is
not representative of the full difficulty distribution.

The benchmark [paper](https://benchmarks.pathfinding.ai/assets/pdf/Nobes2023.pdf)
specifies 26-connected movement and no diagonal passage through filled
corners or edges, but its exact `.3dscen` trailing-column schema and full
evaluation implementation were not verified here. Thus the result is
`technical_diagnostic_not_comparable_to_published_scores`. A point-agent
path is also not a finite-radius pipe or drone route.

On this Windows machine, a fresh process took 18.50 seconds for parse,
validation, `.npy`/world-pack compilation, and 16 searches; its OS-reported
peak working set was 448,647,168 bytes. These are local engineering diagnostics,
not cross-machine performance claims.

Reproduce with the exact source bytes above and:

```powershell
$assets = 'C:\CodeWorkspace\theseo-anysearch\runtime\research-assets'
python -m theseo_anysearch.environments.voxel_benchmark `
  "$assets\plant01.3dmap.zip" `
  "$assets\plant01.3dscen" `
  "$assets\plant01-g1-diagnostic" `
  --query-limit 16
```

The ignored output directory contains `occupancy.npy`, an immutable compiled
world pack, and `diagnostic-report.json` with source-line IDs, selected
task hashes, per-query costs, expansions, replay status, elapsed time, and
memory. Dataset and world identities remain stable across reruns; runtime and
memory are observational values.
