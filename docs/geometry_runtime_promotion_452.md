# Geometry Runtime Promotion (#452)

Base: `develop@b7d746a`. Governing routing roadmap:
[`specs@1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b`](https://github.com/amadou-6e/specs/blob/1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b/projects/theseo-anysearch/routing-geometry-integration-roadmap.md).
Provider contract:
[`specs@84caf74220cc82e68e4d314fe8a98239d93f0927`](https://github.com/amadou-6e/specs/blob/84caf74220cc82e68e4d314fe8a98239d93f0927/projects/theseo-anysearch/world-provider-cli.md).

## Source Mapping

| Issue | Experimental SHA | Promotion |
| --- | --- | --- |
| #296 | `3baa06a` | Compiled-world episode-source guard; preserve newer develop feasibility validation |
| #315 | `b095a6d` | Already on develop as `0ae1f71`; not cherry-picked again |
| #411 | `0f6bc47` | Routing identity, rights, split and task contracts |
| #412 | `b996988` | Standalone point-path benchmark importer |
| #413 | `6d9859d` | Aerial Gym compact collision export |
| #414 | `4194a0f` | Pinned CaveDroneSim geometry exporter |
| #428 | `fde6f11` | Already on develop as `da724a6`; replay files identical |
| #415 | `8f8a87e` | Historical report only; encoder feature evaluator stays on exp |
| #426 | `5a6fccd` | Swept-sphere route export/replay; encoder preparation dependency excluded |
| #429 | `552b1f1` | Generic installable-provider API, fixture, worlds CLI, previews and YAML selection |
| #435 | `65a17a5` | Shared explicit sphere/axis-heading collision checker |

The experimental branch is not merged wholesale into develop. No code is
removed from it, and no frozen study identity is changed. The files
`garden/external_routing.py`, `garden/external_routing_cli.py`, and their
encoder tests remain experimental because they import `CompactEncoder`.
The standalone exporter test keeps its reproducibility and sidecar checks;
the compact preparation assertions remain in the experimental test suite.
The unreviewed optional CaveDrone wheel (#439) is not promoted in this PR.

## Integration Corrections

- Both `geometry` and `worlds` CLI groups remain available.
- Explanation and RLlib Torch-model imports are lazy so the worlds CLI and
  experiment loader work without optional Torch. A subprocess regression test
  blocks Torch imports to enforce this boundary. This focused fix originates
  in #430 (`a1611de`); the provider implementation itself is not included.
- The core wheel includes the repository-owned CaveDrone C++ bridge.
- Remove the mistakenly tracked root `AGENTS.md` and add its ignore rule.
  The user's separately modified root-worktree copy is untouched.

## Validation

Affected non-Ray/non-integration suites: **1,030 passed, 241 deselected**.
This includes the opt-in real-source repeated export with
`CAVEDRONE_SOURCE` pointing to the pinned MIT CaveDroneSim checkout.
One existing usage-config test is excluded locally because it discards any
path containing `runtime`, including this nested issue worktree; it remains
enabled in normal-checkout CI. Other deselections are integration/Ray tests.
`compileall` and `git diff --check` pass. The core wheel built successfully
(SHA-256 `9202d93ad381fffa36a71adab7b3b797003efcfd50c7e2e011fa86c9af5ac28e`).

## Forward Synchronization

After this promotion is accepted, merge current develop forward into
`exp/perception-encoder` through a separate synchronization issue/PR.
Preserve encoder-only files and evidence; resolve the AGENTS deletion using
the owner's explicit local-only guidance. Do not revert geometry code or
force-push either shared branch. New generic routing tasks target develop;
encoder-specific experiments keep their dedicated integration line.
