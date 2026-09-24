# CaveDrone installation

CaveDrone is an optional provider. The core install extra selects the separate
`theseo-anysearch-cavedrone` wheel; it does not bundle upstream simulator assets.
The provider implementation is supplied by PR #439 and must be integrated before
building its wheel from this checkout.

No PyPI publication is required. Build both wheels locally:

```powershell
python -m pip wheel --no-deps --wheel-dir runtime/wheels .
python -m pip wheel --no-deps --wheel-dir runtime/wheels ./providers/cavedrone
python -m pip install --find-links runtime/wheels "theseo-anysearch[cavedrone]==0.1.0"
anysearch worlds list
```

Use `--no-index` on installation when the wheel directory also contains all
transitive dependencies. Installing core alone does not install this provider.
These commands build and install locally; they do not upload packages.

Generation still requires the reviewed upstream checkout configured through
`ANYSEARCH_CAVEDRONE_SOURCE` and a compatible C++ compiler. Adding this install
extra does not implement automatic source downloading or change voxel resolution.
