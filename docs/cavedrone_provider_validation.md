# CaveDrone provider validation (#430)

Governing interface: `amadou-6e/specs@84caf74220cc82e68e4d314fe8a98239d93f0927`.
Route and source contract: `amadou-6e/specs@1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b`.
Disposition requested: `retain`, pending review. No training or GPU run.

## Local installation and execution

- Built separate core and `theseo-anysearch-cavedrone` 0.1.0 wheels, then
  installed both with declared dependencies in a new Python 3.12 virtual
  environment. The optional wheel contains its provider wrapper and entry
  point, not the upstream source, generated worlds, or model weights. The
  core wheel includes the repository-owned C++ bridge source needed at
  generation time. Neither wheel was uploaded to a package index.
- From outside the source worktree, `anysearch worlds list` discovered
  `cavedrone` and advertised native `192x56x192 @ 0.5 m/voxel`.
- With `ANYSEARCH_CAVEDRONE_SOURCE` set to the locally reviewed MIT checkout
  at `ef7852198249390806d8c0cd42e576e02c73c19f` and C++23 `g++`
  available, ran `anysearch worlds cavedrone --seed 42 --output ...` twice.
  Both outputs had two verified tasks, six valid PNG previews, and no rejected
  task strata. The `xy-slice.png` preview was visually inspected.
- Ran `anysearch worlds add ... --config .../train.yaml` against the installed
  wheels. It recorded the verified world/pack/waypoint identities; a repeat
  returned `status: unchanged`. `worlds add` does not attach a test world.
- Uninstalled the optional wheel: the core CLI still listed its fixture
  provider. Reinstalled the optional wheel: `cavedrone` appeared again. The
  final rebuilt core wheel also generated seed 42 with the same verification
  hash, including the declared-split rights check.
- `--partition test --body-radius-m 0.25` generated a separate verified
  test-split bundle for seed 42. Attempting `worlds add` on that bundle with a
  training YAML failed with `world split record does not assign it to the
  requested role`.

## Reproducibility anchors

| Item | SHA-256 / identity |
| --- | --- |
| Pinned upstream revision | `ef7852198249390806d8c0cd42e576e02c73c19f` |
| Source content identity | `f0ae20d61d68a9df9d2a6aed85813dbc02c1fc29bebe591842565a3ecee15fdb` |
| Occupancy NPY | `f474cc14c44fbef2d153504512cf311fecff2a908e303d44da870e9f832ba7bf` |
| World identity | `f265ef0449de39a4b1af61cfa874d7c7228ddf6c4ebe0eea41f3df15eecf15ee` |
| Dataset identity | `6a9567fc012516d3eac99fcd7aa30b6300fa46ffba3022970b6846934a9bf1ab` |
| All three `verification.json` files | `cdadfc2798b13633d76200b7f74c3f4748b2b5ff608b22198cc333c38e6025b1` |
| Built core wheel | `d87a12d096396cf38e8c7f90c7be51b9a02398bd64820dd90e8f6d33351c4f67` |
| Built optional provider wheel | `d6df9a8d3033178bd9e5ec5b78cf9368803efda39459b0f1b1904ca77750519d` |

Ignored local inspection outputs are `runtime/output/cavedrone430-clean-seed42`,
`runtime/output/cavedrone430-clean-seed42-repeat`, and
`runtime/output/cavedrone430-final-seed42`. Their source,
occupancy, task/reference, preview, and verification file hashes matched.

## Tests and limits

The affected suite passed: **350 passed, one deselected** (an existing test
assumes a top-level checkout and fails in nested issue worktrees). This run
included both pinned-source export tests. `compileall` and `git diff --check`
were clean. Offline tests cover missing source, invalid parameters, zero
accepted tasks, mismatched native extent, shared-study split leakage, and an
unrelated YAML sequence in the study directory, and source-rights rejection
for a declared training split.

The generated occupancy is full truth, not a sensor observation. Derived
six-axis swept-sphere voxel routes are not native CaveDroneSim exploration,
continuous controller validation, or path optimality. Seed changes are
within the single `cavedronesim_native_chamber_tunnel_v1` family; no
cross-family generalization claim follows from this validation.
