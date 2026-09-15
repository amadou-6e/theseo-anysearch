"""Generate, verify, index, and rediscover provider worlds."""

from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

from theseo_anysearch.cli.registry import _repo_root_from
from theseo_anysearch.world_providers.api import GenerationSummary, ProviderInfo, load_provider
from theseo_anysearch.world_providers.bundle import VerifiedBundle, load_bundle, sha256
from theseo_anysearch.world_providers.render import render_previews

SPEC_SHA = "84caf74220cc82e68e4d314fe8a98239d93f0927"
_VERIFY_FILES = ("occupancy.npy", "source.json", "conversion.json", "world.json", "split.json")


def registry_path() -> Path:
    override = os.getenv("ANYSEARCH_WORLDS_REGISTRY")
    if override:
        return Path(override)
    root = _repo_root_from(Path.cwd())
    return (root if root is not None else Path.home()) / ".anysearch" / "worlds.json"


def local_worlds() -> list[dict]:
    path = registry_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("world registry must be a JSON list")
    return [row for row in data if isinstance(row, dict)]


def _register_world(root: Path, bundle: VerifiedBundle) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in local_worlds() if row.get("path") != str(root.resolve())]
    rows.append({
        "path": str(root.resolve()),
        "world_identity_sha256": bundle.world.identity_sha256,
        "root_geometry_id": bundle.world.root_geometry_id,
    })
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def remote_catalog() -> list[dict]:
    """Read an explicitly configured versioned catalog; never install packages."""

    value = os.getenv("ANYSEARCH_WORLDS_CATALOG")
    if value and value.startswith("https://"):
        with urlopen(value, timeout=8) as response:
            encoded = response.read(1024 * 1024 + 1)
        if len(encoded) > 1024 * 1024:
            raise ValueError("remote provider catalog is too large")
        data = json.loads(encoded)
    elif value and "://" in value:
        raise ValueError("remote provider catalog URL must use HTTPS")
    elif value:
        data = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        data = json.loads(files("theseo_anysearch.world_providers").joinpath("catalog.json").read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("providers"), list):
        raise ValueError("invalid remote provider catalog")
    result = []
    for row in data["providers"]:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("name", "distribution", "version")):
            raise ValueError("invalid remote provider catalog entry")
        result.append(row)
    return result


def _parameters(info: ProviderInfo, values: dict[str, object]) -> dict[str, object]:
    allowed = {item.name: item for item in info.parameters}
    extra = set(values) - set(allowed)
    if extra:
        raise ValueError(f"unknown {info.name} parameters: {', '.join(sorted(extra))}")
    result = {}
    for name, item in allowed.items():
        if name in values:
            result[name] = item.validate(values[name])
        elif item.default is not None:
            result[name] = item.validate(item.default)
        elif item.required:
            raise ValueError(f"missing required parameter: {name}")
    return result


def _file_hashes(root: Path, bundle: VerifiedBundle) -> dict[str, str]:
    names = [*_VERIFY_FILES]
    names.extend(path.name for path in sorted(root.glob("task-*.json")))
    names.extend(path.name for path in sorted(root.glob("reference-*.json")))
    for reference in bundle.references:
        assert reference.route_artifact is not None
        names.append(reference.route_artifact.relative_path)
    return {name: sha256(root / name) for name in sorted(set(names))}


def generate_world(name: str, *, seed: int, output: Path, parameters: dict[str, object] | None = None) -> dict:
    provider = load_provider(name)
    if output.exists():
        raise FileExistsError(f"world output already exists: {output}")
    resolved = _parameters(provider.info, parameters or {})
    summary = provider.generate(seed=seed, output=output, parameters=resolved)
    if not isinstance(summary, GenerationSummary):
        raise ValueError("provider must return a GenerationSummary")
    bundle = load_bundle(output)
    if bundle.world.frame.meters_per_voxel != provider.info.output_resolution(resolved):
        raise ValueError("provider output resolution disagrees with its declared native resolution")
    previews = render_previews(bundle)
    report = {
        "schema_version": 1,
        "governing_spec_sha": SPEC_SHA,
        "provider": name,
        "provider_version": provider.info.version,
        "parameters": {"seed": seed, **resolved},
        "rejected_task_strata": list(summary.rejected_task_strata),
        "world_identity_sha256": bundle.world.identity_sha256,
        "dataset_identity_sha256": bundle.dataset_identity_sha256,
        "files": _file_hashes(bundle.root, bundle),
        "previews": previews,
        "verification": "all task endpoints and six-axis swept-sphere routes replayed against complete occupied voxel cubes; not continuous controller feasibility or optimality",
    }
    (bundle.root / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _register_world(bundle.root, bundle)
    return report


def load_verified_world(root: Path, *, use: str | None = None) -> VerifiedBundle:
    bundle = load_bundle(root, use=use)
    report = json.loads((bundle.root / "verification.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("world_identity_sha256") != bundle.world.identity_sha256:
        raise ValueError("world verification report disagrees with its manifest")
    if report.get("dataset_identity_sha256") != bundle.dataset_identity_sha256:
        raise ValueError("world dataset identity changed after verification")
    expected = _file_hashes(bundle.root, bundle)
    if report.get("files") != expected:
        raise ValueError("verified world files changed after generation")
    previews = report.get("previews")
    expected_previews = {
        f"previews/{label}-{mode}.png"
        for label in ("xy", "xz", "yz") for mode in ("projection", "slice")
    }
    if not isinstance(previews, dict) or set(previews) != expected_previews:
        raise ValueError("world has no complete PNG preview set")
    for name, digest in previews.items():
        if sha256(bundle.root / name) != digest:
            raise ValueError("world PNG preview changed after verification")
    return bundle
