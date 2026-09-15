# Gazebo adapter integration (#466)

Governing successor spec:
`amadou-6e/specs@d98320117af66e3189d2b1b83b8f49897e91cfd2`
(`projects/theseo-anysearch/gazebo-world-provider.md`, specs PR #98).
Explicit owner authorization permits stacked implementation before review;
neither specification nor implementation is merged automatically.

Selectively ports `d9bcc981f4bd775c3ebb5e0ac7d0732fc3a5f569` from #417/#424:
only the converter, its tests and frozen historical report. No experimental
history, encoder modules or AGENTS.md changes are imported. Historical v1
evidence remains unchanged. The adapter additionally rejects unsafe archive
members and bounds decompressed bytes/member count and voxel allocations.

Offline adapter suite: 15 passed. Includes transform composition, missing
dependencies, pinned hash rejection, conservative primitive rasterization,
roof closure, source replay, invalid resolution and archive attack cases.
Disposition: retain shared infrastructure; no training or package publication.

Native Gazebo inspection exposed and #477 records a blocking pre-review defect:
Gazebo is source-Z-up while the replayer is storage-Y-up. Storage now maps to
source `(x,z,y)` through `GridFrame`; rasterization, source parity, planar
routing, roof census and altitude metrics use that same mapping. Pre-fix packs
and their identities are invalid viewer evidence and are superseded.
