# Aerial Gym provider validation (#457)

Governing provider spec: `amadou-6e/specs@6e4f2d7d43cb8f90e14d9e68cd1df1d492c66a6d`
(specs PR #96). Existing route contract: `1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b`.
Foundation: #459 / PR #460, stacked on `develop`; no merge authorized.

Built both distributions and installed the core `aerialgym` extra using
`pip --no-index --find-links` against local wheels in the previously isolated
installed-wheel validation environment. Dependency resolution installed the
separate optional provider. No package-index publication or training occurred.

First-use generation downloaded ten hash-pinned source files from upstream
`f0d0f05283f7897bab5a1bcc7b19b91cebbab218`. All downloads matched the allowlist.
An initial rights-enum integration error occurred after caching, before world
publication; corrected to the existing training/evaluation/redistribution enum.
Subsequent runs used `--offline true`. Source caches and world outputs remain
ignored under `runtime/output`; neither is committed or bundled in wheels.

| Run | Voxel grid | World identity |
| --- | --- | --- |
| seed 42 detour, 0.25 m | 40 x 40 x 24 | `9e8b0871036cd438f1d130c0b795aa01e84c3391fc538c6c8000b945c1957a25` |
| seed 42 detour, 0.10 m | 100 x 100 x 60 | `718ead89747e7c7735cb4b2dff0f923ff09807fb28d48d8a53d904d7b848ef1e` |
| installed wheels, seed 42 detour, 0.10 m | 100 x 100 x 60 | same as above |
| installed wheels, seed 42 altitude, 0.25 m | 40 x 40 x 24 | `c9b34e118f727f9811130974761472b527fc67ab8455503681cb42f3f8dce07c` |

All four successful runs produced one independently checked task and six PNGs.
The detour scene-instance SHA-256 was identical at both resolutions:
`4a18f9539726f69f5548017b8fd10907c4b091e8e473f60320c77c534ea085f8`.
The source-checkout and installed-wheel 0.10 m verification reports matched:
`e6c9ef322086d5a6c55cf020a5353876f848640d078385f5a36f41510291fb46`.
The altitude route has 50 six-axis steps vs 24 direct steps, and no same-altitude
route. Both detour and altitude PNGs were visually inspected.

Installed `worlds add` compiled the 0.25 m grid and updated an isolated existing
training YAML with selected task
`abb0773e279f52cd4fd963ca7de3c8c200b76c503f9b2508b6b4e7c2ee2af7c0`
and world-pack identity
`fcd4cb567e6eddf90a542df0a73b15daa331da8ae0dea29f6538763f004b3f56`.

The focused provider/export suite passed 39 tests; the broader affected
provider/environment unit suite passed 198 tests with one skipped.
An independent test fixture
identified that source-box clearance alone is insufficient for conservatively
rasterized voxel-cube replay. The provider now additionally blocks a conservative
separable voxel dilation before searching, while retaining independent source-box
validation and core swept-sphere replay. This is deliberately conservative, not
a claim of shortest continuous routes. Existing exporter defaults are preserved.

Rights review applies only to the allowlisted rigid box URDFs and source configs:
no conflicting headers or referenced meshes, with upstream BSD-3-Clause LICENSE
retained and hash-checked in every generated bundle. It does not clear arbitrary
upstream assets. Native Isaac Gym parity remains #423; no sensor/controller,
cross-family generalization or model-training claim is made here.
