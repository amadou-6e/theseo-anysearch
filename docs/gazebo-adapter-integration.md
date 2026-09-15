# Gazebo adapter integration (#466)

Governing successor spec:
`amadou-6e/specs@d54d03501a9fc5b6f8be126c4ac6e8439c84882c`
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
