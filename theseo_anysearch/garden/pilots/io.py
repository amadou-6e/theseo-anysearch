"""Canonical, content-addressed IO for frozen pilot contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from theseo_anysearch.garden.pilots.contracts import model_payload


ContractT = TypeVar("ContractT", bound=BaseModel)


class ContractIntegrityError(ValueError):
    """Raised when a stored contract does not match its declared identity."""


class FrozenArtifactError(FileExistsError):
    """Raised when code attempts to replace an already frozen contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def contract_sha256(contract: BaseModel) -> str:
    """Return the stable identity of a validated contract payload."""

    return hashlib.sha256(_canonical_json(model_payload(contract))).hexdigest()


def _envelope(contract: BaseModel) -> dict[str, Any]:
    payload = model_payload(contract)
    return {
        "envelope_version": 1,
        "kind": type(contract).__name__,
        "identity_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        "payload": payload,
    }


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ContractIntegrityError("contract envelope must be a mapping")
    return raw


def write_contract(path: Path, contract: BaseModel) -> str:
    """Write a frozen contract once, allowing only an identical idempotent write."""

    path = Path(path)
    envelope = _envelope(contract)
    if path.exists():
        existing = _load_raw(path)
        if existing != envelope:
            raise FrozenArtifactError(f"refusing to replace frozen contract: {path}")
        return envelope["identity_sha256"]

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        serialized = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    elif path.suffix.lower() in {".yaml", ".yml"}:
        serialized = yaml.safe_dump(envelope, sort_keys=True)
    else:
        raise ValueError("pilot contracts must use .json, .yaml, or .yml")
    path.write_text(serialized, encoding="utf-8", newline="\n")
    return envelope["identity_sha256"]


def read_contract(path: Path, contract_type: type[ContractT]) -> ContractT:
    """Load and verify a typed frozen contract."""

    raw = _load_raw(Path(path))
    required = {"envelope_version", "kind", "identity_sha256", "payload"}
    if set(raw) != required or raw.get("envelope_version") != 1:
        raise ContractIntegrityError("invalid contract envelope")
    if raw["kind"] != contract_type.__name__:
        raise ContractIntegrityError(
            f"expected contract kind {contract_type.__name__}, found {raw['kind']}"
        )
    actual = hashlib.sha256(_canonical_json(raw["payload"])).hexdigest()
    if actual != raw["identity_sha256"]:
        raise ContractIntegrityError("contract payload does not match identity_sha256")
    return contract_type.model_validate(raw["payload"])
