"""Compile, archive, validate, and execute experiment-local Rust extensions."""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

ABI_VERSION = 2
CAP_REWARD = 1
CAP_TRAINING_METRICS = 2
CAP_EVALUATION_METRICS = 4
METRIC_BUFFER_SIZE = 65536


class NativeExtensionError(RuntimeError):
    """Raised when a native extension cannot be compiled or violates the ABI."""


class NativeExtensionManifest(BaseModel):
    """Portable metadata for one compiled extension artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    abi_version: int
    source_sha256: str
    binary_sha256: str
    library: str
    capabilities: tuple[str, ...]
    rewards: tuple[str, ...] = ()
    platform: str
    machine: str


def _extension_sdk_dir() -> Path:
    sdk = Path(__file__).resolve().parent.parent.joinpath("extension_sdk")
    if not sdk.joinpath("Cargo.toml").is_file():
        raise NativeExtensionError(f"Bundled Rust extension SDK is missing: {sdk}")
    return sdk


def _cargo_command(command: str, cargo_manifest: Path, *arguments: str) -> list[str]:
    sdk = _extension_sdk_dir().as_posix()
    patch = f'patch.crates-io.anysearch-extension.path="{sdk}"'
    return [
        "cargo",
        "--config",
        patch,
        command,
        "--manifest-path",
        str(cargo_manifest),
        *arguments,
    ]


def _source_digest(extension_dir: Path) -> str:
    digest = hashlib.sha256()
    roots = (("extension", extension_dir), ("sdk", _extension_sdk_dir()))
    for label, root in roots:
        files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and "target" not in path.relative_to(root).parts
        )
        for path in files:
            relative = f"{label}/{path.relative_to(root).as_posix()}".encode()
            digest.update(len(relative).to_bytes(8, "little"))
            digest.update(relative)
            data = path.read_bytes()
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
    return digest.hexdigest()

def _library_filename(crate_name: str) -> str:
    stem = crate_name.replace("-", "_")
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def _selected_reward_name(experiment_dir: Path) -> str | None:
    import yaml

    candidates = [
        experiment_dir.joinpath("experiment.yaml"),
        experiment_dir.joinpath("config.yaml"),
    ]
    candidates.extend(sorted(experiment_dir.glob("*.yaml")))
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        return None
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rewards = (raw.get("env") or {}).get("rewards") or {}
    selected = rewards.get("provider", rewards.get("custom"))
    if isinstance(selected, str) or selected is None:
        return selected
    if isinstance(selected, dict):
        name = selected.get("name")
        return str(name) if name is not None else None
    return None

def compile_native_extension(experiment_dir: Path, *, force: bool = False) -> Path:
    """Compile ``extension/`` and return its stable manifest path."""
    root = experiment_dir.resolve()
    extension_dir = root.joinpath("extension")
    cargo_manifest = extension_dir.joinpath("Cargo.toml")
    if not cargo_manifest.is_file():
        raise NativeExtensionError(f"Missing {cargo_manifest}")
    cargo = tomllib.loads(cargo_manifest.read_text(encoding="utf-8"))
    crate_name = str(cargo.get("lib", {}).get("name") or cargo.get("package", {}).get("name", ""))
    if not crate_name:
        raise NativeExtensionError("Cargo.toml must define package.name or lib.name")
    subprocess.run(
        _cargo_command("generate-lockfile", cargo_manifest),
        cwd=extension_dir, check=True,
    )
    source_sha = _source_digest(extension_dir)
    reward_name = _selected_reward_name(root)
    build_dir = root.joinpath(".anysearch", "build", source_sha)
    stable_manifest = root.joinpath(".anysearch", "extension.json")
    if stable_manifest.is_file() and not force:
        current = NativeExtensionManifest.model_validate_json(stable_manifest.read_text(encoding="utf-8"))
        candidate = stable_manifest.parent.joinpath(current.library)
        expected_rewards = (reward_name,) if "reward" in current.capabilities and reward_name else ()
        if (
            current.source_sha256 == source_sha
            and current.rewards == expected_rewards
            and candidate.is_file()
        ):
            loaded = NativeExtension.load(stable_manifest)
            if loaded is not None and reward_name is not None:
                loaded.validate_reward(reward_name)
            return stable_manifest
    subprocess.run(
        _cargo_command("build", cargo_manifest, "--release"),
        cwd=extension_dir, check=True,
    )
    filename = _library_filename(crate_name)
    built = extension_dir.joinpath("target", "release", filename)
    if not built.is_file():
        raise NativeExtensionError(f"Cargo did not produce expected cdylib: {built}")
    build_dir.mkdir(parents=True, exist_ok=True)
    target = build_dir.joinpath(filename)
    shutil.copy2(built, target)
    temporary = build_dir.joinpath("extension.json")
    relative = target.relative_to(stable_manifest.parent).as_posix()
    probe = NativeExtension.load_library(target)
    capabilities = probe.capability_names
    rewards: tuple[str, ...] = ()
    if "reward" in capabilities:
        if reward_name is None:
            raise NativeExtensionError(
                "A reward-capable extension requires env.rewards.provider in YAML"
            )
        probe.validate_reward(reward_name)
        rewards = (reward_name,)
    manifest = NativeExtensionManifest(
        abi_version=ABI_VERSION, source_sha256=source_sha,
        binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        library=relative, capabilities=capabilities, rewards=rewards,
        platform=sys.platform, machine=platform.machine(),
    )
    stable_content = manifest.model_dump_json(indent=2)
    build_manifest = manifest.model_copy(update={"library": target.name})
    temporary.write_text(build_manifest.model_dump_json(indent=2), encoding="utf-8")
    stable_manifest.parent.mkdir(parents=True, exist_ok=True)
    stable_manifest.write_text(stable_content, encoding="utf-8")
    return stable_manifest


def discover_native_manifest(config_path: Path | None) -> Path | None:
    """Find and reject a stale compiled extension beside an experiment YAML."""
    if config_path is None:
        return None
    manifest_path = config_path.parent.joinpath(".anysearch", "extension.json")
    if not manifest_path.is_file():
        return None
    manifest = NativeExtensionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    selected_reward = _selected_reward_name(config_path.parent)
    if "reward" in manifest.capabilities and manifest.rewards != ((selected_reward,) if selected_reward else ()):
        raise NativeExtensionError(
            "Selected YAML reward changed; run "
            f"'anysearch compile {config_path.parent}'"
        )
    extension_dir = config_path.parent.joinpath("extension")
    if not extension_dir.is_dir() or _source_digest(extension_dir) != manifest.source_sha256:
        raise NativeExtensionError(
            f"Native extension sources changed; run 'anysearch compile {config_path.parent}'"
        )
    NativeExtension.load(manifest_path)
    return manifest_path


def copy_native_extension(config_path: Path | None, destination: Path) -> Path | None:
    """Archive the exact validated manifest and binary used by a run."""
    source_manifest = discover_native_manifest(config_path)
    if source_manifest is None:
        return None
    manifest = NativeExtensionManifest.model_validate_json(source_manifest.read_text(encoding="utf-8"))
    source_binary = source_manifest.parent.joinpath(manifest.library)
    native_dir = destination.joinpath("native_extension")
    native_dir.mkdir(parents=True, exist_ok=True)
    target_binary = native_dir.joinpath(source_binary.name)
    shutil.copy2(source_binary, target_binary)
    archived = manifest.model_copy(update={"library": target_binary.name})
    target_manifest = native_dir.joinpath("extension.json")
    target_manifest.write_text(archived.model_dump_json(indent=2), encoding="utf-8")
    NativeExtension.load(target_manifest)
    return target_manifest


class NativeExtension:
    """Validated handle to the stable AnySearch native extension ABI v2."""

    def __init__(self, library: Any, source: Path, capabilities: int) -> None:
        self._library = library
        self.source = source
        self.capabilities = capabilities

    @property
    def capability_names(self) -> tuple[str, ...]:
        pairs = ((CAP_REWARD, "reward"), (CAP_TRAINING_METRICS, "training_metrics"),
                 (CAP_EVALUATION_METRICS, "evaluation_metrics"))
        return tuple(name for flag, name in pairs if self.capabilities & flag)

    def validate_reward(self, name: str) -> None:
        if not name.isidentifier():
            raise NativeExtensionError(f"Invalid reward name {name!r}")
        symbol = f"anysearch_reward_{name}_v2"
        try:
            getattr(self._library, symbol)
        except AttributeError as exc:
            raise NativeExtensionError(
                f"Native extension does not export selected reward {name!r} ({symbol})"
            ) from exc

    @classmethod
    def load_library(cls, path: Path) -> "NativeExtension":
        library = ctypes.CDLL(str(path))
        try:
            library.anysearch_extension_abi_version.restype = ctypes.c_uint32
            library.anysearch_extension_capabilities.restype = ctypes.c_uint64
            version = int(library.anysearch_extension_abi_version())
            capabilities = int(library.anysearch_extension_capabilities())
        except AttributeError as exc:
            raise NativeExtensionError(f"{path} does not export the AnySearch ABI") from exc
        if version != ABI_VERSION:
            raise NativeExtensionError(f"{path}: ABI version {version}, expected {ABI_VERSION}")
        return cls(library, path, capabilities)

    @classmethod
    def load(cls, manifest_path: Path | None) -> "NativeExtension | None":
        if manifest_path is None:
            return None
        manifest = NativeExtensionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.abi_version != ABI_VERSION:
            raise NativeExtensionError(
                f"Manifest ABI version {manifest.abi_version}, expected {ABI_VERSION}"
            )
        if manifest.platform != sys.platform or manifest.machine != platform.machine():
            raise NativeExtensionError(
                f"Native extension targets {manifest.platform}/{manifest.machine}, "
                f"not {sys.platform}/{platform.machine()}"
            )
        binary = manifest_path.parent.joinpath(manifest.library)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest.binary_sha256:
            raise NativeExtensionError(f"Native extension binary hash mismatch: {binary}")
        loaded = cls.load_library(binary)
        if set(loaded.capability_names) != set(manifest.capabilities):
            raise NativeExtensionError("Native extension capabilities do not match its manifest")
        return loaded

    def compute_metrics(self, scope: str, context: Mapping[str, Any]) -> dict[str, float]:
        flag = CAP_TRAINING_METRICS if scope == "training" else CAP_EVALUATION_METRICS
        if not self.capabilities & flag:
            return {}
        payload = json.dumps(context, separators=(",", ":"), default=str).encode()
        output = ctypes.create_string_buffer(METRIC_BUFFER_SIZE)
        length = ctypes.c_size_t()
        function = getattr(self._library, f"anysearch_compute_{scope}_metrics_v1")
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                             ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        function.restype = ctypes.c_int32
        status = int(function(payload, len(payload), output, len(output), ctypes.byref(length)))
        if status != 0 or length.value >= len(output):
            raise NativeExtensionError(f"Native {scope} metrics failed with status {status}")
        raw = json.loads(output.raw[:length.value])
        if not isinstance(raw, dict):
            raise NativeExtensionError(f"Native {scope} metrics must return a JSON object")
        return {str(name): float(value) for name, value in raw.items()}


def validate_native_metrics(
    scope: str,
    metrics: Mapping[str, float],
    *,
    reserved_names: set[str],
) -> dict[str, float]:
    """Validate and scope metric names returned by native code."""
    import math

    validated: dict[str, float] = {}
    for name, value in metrics.items():
        if not name.isidentifier() or isinstance(value, bool) or not math.isfinite(value):
            raise NativeExtensionError(f"Invalid native {scope} metric {name!r}")
        full_name = f"{scope}_{name}"
        if full_name in reserved_names or full_name in validated:
            raise NativeExtensionError(f"Native metric {full_name!r} is reserved or duplicated")
        validated[full_name] = value
    return validated
