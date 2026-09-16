# IFC-Bench MEP voxel provider

Install the core extra: `python -m pip install "theseo-anysearch[ifcbench]"`.
Until release, build both wheels and install the extra with pip `--find-links`
pointing to the local wheel directory. Installing pip packages never downloads
assets or generates worlds.

```powershell
anysearch worlds list
anysearch worlds ifcbench --seed 0 --discipline plumbing --meters-per-voxel 0.01 --output worlds/wrh-plumb-0
anysearch worlds ifcbench --seed 0 --discipline electrical --meters-per-voxel 0.013949999999999999 --output worlds/wrh-elec-0
anysearch worlds add worlds/wrh-plumb-0 --config experiments/train.yaml
```

Generation downloads the two pinned West Riverside Hospital IFC files
(`plumb_ifc4.ifc`, `elec_ifc4.ifc`) from `sylvainHellin/ifc-bench` at tagged
release v2.0.1, not the whole dataset. Cache location:
`ANYSEARCH_IFCBENCH_CACHE` or `~/.cache/anysearch/ifcbench`. Every reuse
verifies source hashes. Offline mode refuses an empty cache.

`--meters-per-voxel` is not a free resampling choice: each discipline is
extracted from a fixed, non-user-selectable canonical crop of the real
building (a real discipline spans roughly 85 x 33 x 65 m -- far past this
adapter's voxel cap at pipe-appropriate resolution), and the resolution
below follows from that crop's own real geometry. `generate_world`'s
declared-vs-actual check is an exact floating-point comparison, so the
value must be typed exactly as shown (electrical is not resamplable to a
round decimal; it is `min(radius)/2` on the real pinned file):

| Discipline | meters-per-voxel |
| --- | --- |
| plumbing | 0.01 |
| electrical | 0.013949999999999999 |

`--body-radius-m` defaults to 0.02 and lies in [0, 0.5]. Every accepted task
is independently reverified twice: as a swept sphere against the quantized
occupied voxel grid, and as a continuous-space segment against every
extracted element's own AABB -- not against the exact triangulated solid or
against unextracted structural/architectural/mechanical/fire/sprinkler IFC
content for the same building. Seed rotates the fixed accepted-task roster's
order; it does not alter which tasks are accepted or the static source
geometry. The model's own `license.txt` (CC BY 3.0 Unported, distinct from
the Hugging Face dataset card's repository-level CC BY 4.0 tag) is preserved
in each bundle.

Current CLI attaches one world and one selected task per training YAML;
generation does not start training.

Governing merged spec: `amadou-6e/specs@adcf97b3312d2cd4d1fd3cf9be85441559d2c5f9`
(`projects/theseo-anysearch/ifc-bench-mep-provider.md`).
