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

from theseo_anysearch.experiments.custom_rewards import RewardContext, RewardResult

ABI_VERSION = 1
CAP_REWARD = 1
CAP_TRAINING_METRICS = 2
CAP_EVALUATION_METRICS = 4
MAX_COMPONENTS = 8
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
    platform: str
    machine: str


class _RewardContextV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("step", ctypes.c_uint64), ("action_index", ctypes.c_int32),
        ("previous_cursor", ctypes.c_int32 * 3), ("cursor", ctypes.c_int32 * 3),
        ("goal", ctypes.c_int32 * 3), ("has_goal", ctypes.c_uint8),
        ("invalid_action", ctypes.c_uint8), ("collision", ctypes.c_uint8),
        ("terminated", ctypes.c_uint8), ("truncated", ctypes.c_uint8),
        ("previous_goal_distance", ctypes.c_double),
        ("goal_distance", ctypes.c_double), ("standard_reward", ctypes.c_double),
    ]


class _RewardComponentV1(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 64), ("value", ctypes.c_double)]


class _RewardResultV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("mode", ctypes.c_uint32), ("reward", ctypes.c_double),
        ("component_count", ctypes.c_uint32),
        ("components", _RewardComponentV1 * MAX_COMPONENTS),
    ]


def _source_digest(extension_dir: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in extension_dir.rglob("*")
        if path.is_file() and "target" not in path.relative_to(extension_dir).parts
    )
    for path in files:
        relative = path.relative_to(extension_dir).as_posix().encode()
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
        ["cargo", "generate-lockfile", "--manifest-path", str(cargo_manifest)],
        cwd=extension_dir, check=True,
    )
    source_sha = _source_digest(extension_dir)
    build_dir = root.joinpath(".anysearch", "build", source_sha)
    stable_manifest = root.joinpath(".anysearch", "extension.json")
    if stable_manifest.is_file() and not force:
        current = NativeExtensionManifest.model_validate_json(stable_manifest.read_text(encoding="utf-8"))
        candidate = stable_manifest.parent.joinpath(current.library)
        if current.source_sha256 == source_sha and candidate.is_file():
            NativeExtension.load(stable_manifest)
            return stable_manifest
    subprocess.run(
        ["cargo", "build", "--manifest-path", str(cargo_manifest), "--release"],
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
    manifest = NativeExtensionManifest(
        abi_version=ABI_VERSION, source_sha256=source_sha,
        binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        library=relative, capabilities=capabilities,
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
    """Validated handle to the stable AnySearch native extension ABI v1."""

    def __init__(self, library: Any, source: Path, capabilities: int) -> None:
        self._library = library
        self.source = source
        self.capabilities = capabilities

    @property
    def capability_names(self) -> tuple[str, ...]:
        pairs = ((CAP_REWARD, "reward"), (CAP_TRAINING_METRICS, "training_metrics"),
                 (CAP_EVALUATION_METRICS, "evaluation_metrics"))
        return tuple(name for flag, name in pairs if self.capabilities & flag)

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

    def compute_reward(self, context: RewardContext) -> RewardResult:
        if not self.capabilities & CAP_REWARD:
            raise NativeExtensionError("Native extension has no reward capability")
        goal = context.goal or (0, 0, 0)
        raw = _RewardContextV1(
            ABI_VERSION, ctypes.sizeof(_RewardContextV1), context.step, context.action_index,
            (ctypes.c_int32 * 3)(*context.previous_cursor), (ctypes.c_int32 * 3)(*context.cursor),
            (ctypes.c_int32 * 3)(*goal), int(context.goal is not None), int(context.invalid_action),
            int(context.collision), int(context.terminated), int(context.truncated),
            context.previous_goal_distance, context.goal_distance, context.standard_reward,
        )
        result = _RewardResultV1(ABI_VERSION, ctypes.sizeof(_RewardResultV1))
        function = self._library.anysearch_compute_reward_v1
        function.argtypes = [ctypes.POINTER(_RewardContextV1), ctypes.POINTER(_RewardResultV1)]
        function.restype = ctypes.c_int32
        status = int(function(ctypes.byref(raw), ctypes.byref(result)))
        if status != 0 or result.component_count > MAX_COMPONENTS:
            raise NativeExtensionError(f"Native reward failed with status {status}")
        if result.mode not in (0, 1):
            raise NativeExtensionError(f"Native reward returned invalid mode {result.mode}")
        components = {}
        for index in range(result.component_count):
            component = result.components[index]
            name = bytes(component.name).split(b"\0", 1)[0].decode("utf-8")
            components[name] = float(component.value)
        return RewardResult(reward=float(result.reward), components=components,
                            mode="replace" if result.mode == 1 else "add")

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


def apply_native_reward(
    extension: NativeExtension | None,
    context: RewardContext,
) -> tuple[float, dict[str, float]]:
    """Apply a native reward result with the same semantics as Python hooks."""
    if extension is None or not extension.capabilities & CAP_REWARD:
        return context.standard_reward, dict(context.standard_breakdown)
    result = extension.compute_reward(context)
    components = dict(result.components) or {"native_reward": result.reward}
    collisions = set(components) & set(context.standard_breakdown)
    if collisions:
        raise NativeExtensionError(
            f"Native reward components collide with built-ins: {', '.join(sorted(collisions))}"
        )
    if result.mode == "replace":
        return result.reward, components
    return context.standard_reward + result.reward, {**context.standard_breakdown, **components}


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
