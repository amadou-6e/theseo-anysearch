"""Versioned, inspectable recipes for re-executing checkpointed policies."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        name = item.name if path.is_file() else item.relative_to(path).as_posix()
        data = item.read_bytes()
        digest.update(len(name).to_bytes(8, "little")); digest.update(name.encode())
        digest.update(len(data).to_bytes(8, "little")); digest.update(data)
    return digest.hexdigest()


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    path: str
    sha256: str


class Overrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world_manifest: str | None = None
    disabled_capabilities: list[str] = Field(default_factory=list)
    replacements: dict[str, str] = Field(default_factory=dict)


class ExecutionRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    scope: Literal["evaluation", "continuation", "fine_tuning"]
    source_run: str
    checkpoint: Artifact
    experiment: Artifact
    checkpoint_state: dict[str, Any]
    policy_contract: dict[str, Any]
    extension: list[Artifact] = Field(default_factory=list)
    extension_bindings: list[str] = Field(default_factory=list)
    overrides: Overrides = Field(default_factory=Overrides)
    provenance_gaps: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExecutionRecipe":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def clone(checkpoint: Path, scope: str) -> ExecutionRecipe:
    checkpoint = checkpoint.resolve()
    run = checkpoint.parent.parent
    config_path = run / "experiment.yaml"
    state_path = checkpoint / "state.json"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("checkpoint requires parent run experiment.yaml and checkpoint state.json")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("env"), dict) or not isinstance(raw.get("training"), dict):
        raise ValueError("checkpoint parent configuration is not one experiment")
    contract = {"training": {key: raw["training"].get(key) for key in ("algorithm", "model")},
                "observation": raw["env"]["observation"], "action": raw["env"]["action"],
                "task": raw["env"].get("task", {}), "rewards": raw["env"].get("rewards", {})}
    extension = []
    ext_dir = run / "native_extension"
    bindings = []
    if (ext_dir / "extension.json").is_file():
        manifest = json.loads((ext_dir / "extension.json").read_text(encoding="utf-8"))
        for plural, singular in (("rewards", "reward"), ("predicates", "predicate"),
                                 ("outcomes", "outcome"), ("scenarios", "scenario"),
                                 ("geometries", "geometry")):
            bindings.extend(f"{singular}:{name}" for name in manifest.get(plural, []))
        for role, path in (("extension_manifest", ext_dir / "extension.json"),
                           ("extension_binary", ext_dir / manifest["library"])):
            extension.append(Artifact(role=role, path=str(path.resolve()), sha256=_sha(path)))
    provenance = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("provenance.json", "source.json", "run.json"):
        path = run / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in ("source_commit", "git_sha", "commit_sha"):
                if payload.get(key): provenance["source_revision"] = str(payload[key])
    gaps = [] if provenance.get("source_revision") else ["source revision is not recorded in this run"]
    return ExecutionRecipe(scope=scope, source_run=str(run.resolve()),
        checkpoint=Artifact(role="rllib_checkpoint", path=str(checkpoint), sha256=_sha(checkpoint)),
        experiment=Artifact(role="resolved_experiment", path=str(config_path.resolve()), sha256=_sha(config_path)),
        checkpoint_state=json.loads(state_path.read_text(encoding="utf-8")), policy_contract=contract,
        extension=extension, extension_bindings=bindings, provenance_gaps=gaps, provenance=provenance)


def make_portable(recipe: ExecutionRecipe, directory: Path) -> ExecutionRecipe:
    """Copy immutable inputs into a new, self-verifying relocatable bundle."""
    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"bundle directory is not empty: {directory}")
    artifacts = directory / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, artifact in enumerate([recipe.checkpoint, recipe.experiment, *recipe.extension]):
        source = Path(artifact.path)
        target = artifacts / f"{index:02d}-{artifact.role}"
        if source.is_dir(): shutil.copytree(source, target)
        else: shutil.copy2(source, target)
        copied.append(Artifact(role=artifact.role, path=target.relative_to(directory).as_posix(), sha256=_sha(target)))
    return recipe.model_copy(update={"checkpoint": copied[0], "experiment": copied[1],
                                     "extension": copied[2:]})


def validate(recipe: ExecutionRecipe, world: Path | None = None, base: Path | None = None) -> dict[str, Any]:
    artifacts = [recipe.checkpoint, recipe.experiment, *recipe.extension]
    verified = []
    for artifact in artifacts:
        path = Path(artifact.path)
        if not path.is_absolute(): path = (base or Path.cwd()) / path
        if not path.exists():
            raise FileNotFoundError(f"missing {artifact.role}: {path}")
        if _sha(path) != artifact.sha256:
            raise ValueError(f"tampered {artifact.role}: {path}")
        verified.append(artifact.role)
    effective_world = world or (Path(recipe.overrides.world_manifest) if recipe.overrides.world_manifest else None)
    changes = []
    if effective_world:
        manifest = json.loads(effective_world.read_text(encoding="utf-8"))
        old = recipe.checkpoint_state.get("world_contract") or {}
        new_extent = manifest.get("extent")
        if new_extent and old.get("extent") and list(new_extent) != list(old["extent"]):
            changes.append({"component": "world.extent", "from": old["extent"], "to": new_extent})
        changes.append({"component": "world", "from": old.get("identity_sha256") or old.get("catalog_identity_sha256"),
                        "to": manifest.get("identity_sha256"), "manifest": str(effective_world.resolve())})
    if recipe.overrides.disabled_capabilities:
        unknown = [c for c in recipe.overrides.disabled_capabilities if c not in recipe.extension_bindings]
        if unknown:
            raise ValueError(f"disabled extension bindings do not exist: {unknown}")
        missing = [c for c in recipe.overrides.disabled_capabilities if c not in recipe.overrides.replacements]
        if missing:
            raise ValueError(f"disabled capabilities require explicit replacements: {missing}")
    return {"valid": True, "scope": recipe.scope, "verified": verified, "changes": changes,
            "inactive": (["learner", "optimizer", "exploration", "curriculum_adaptation"]
                         if recipe.scope == "evaluation" else []),
            "provenance_gaps": recipe.provenance_gaps,
            "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
            "execution_supported": False,
            "execution_blocker": "this implementation slice validates recipes; policy execution is not enabled yet"}
