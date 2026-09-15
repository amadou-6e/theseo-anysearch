# Benchmark Voxel Maps

These maps are **not bundled** in the repo (files are 200 KB – 4+ MB zipped).
Download what you need and place the `.3dmap.zip` files in this directory.

## Sources

| Set | Maps | Dimensions | URL |
|---|---|---|---|
| Industrial plants | 5 | 195×128×100 – 1257×457×304 | https://bitbucket.org/shortestpathlab/benchmarks/src/master/voxel-maps/industrial-plants/ |
| Descent levels | 27 | up to 1401×678×1385 | https://bitbucket.org/shortestpathlab/benchmarks/src/master/voxel-maps/descent/ |
| Warframe | 40 | ~105–1108 per axis | https://www.movingai.com/benchmarks/warframe/index.html |

**Start with industrial plants** — they're small enough to load fully and dense
enough to be interesting routing environments.

## Quick download (curl / wget)

```bash
# Industrial plants (all 5, ~8 MB total)
BASE=https://bitbucket.org/shortestpathlab/benchmarks/raw/master/voxel-maps/industrial-plants/map_files
for i in 01 02 03 04 05; do
  curl -L "$BASE/plant$i.3dmap.zip" -o "usage/maps/plant$i.3dmap.zip"
done
```

## File format

Plain ASCII text (inside the zip):

```
voxel W H D          # obstacle voxels listed
x y z
x y z
...
```

or

```
rev_voxel W H D      # free-space voxels listed (Warframe maps)
x y z
...
```

## Using in YAML config

```yaml
env:
  map_path: usage/maps/plant01.3dmap.zip
  map_crop_origin: [50, 30, 20]   # [x, y, z] corner in map coords
  grid_size: 32                   # crop window size
  agent_count: 4
  max_steps: 200
  trail_mode: true
  obs_mode: scalar
```

`map_crop_origin` defaults to `[0, 0, 0]` if omitted.  For large maps, use
`suggest_crop_origin` from `theseo_anysearch.environments.map3d` to find a
region with non-trivial obstacle density automatically:

```python
from theseo_anysearch.environments.map3d import suggest_crop_origin

origin = suggest_crop_origin("usage/maps/plant02.3dmap.zip", grid_size=32)
print(origin)  # e.g. (48, 16, 32)
```
