"""Resolved request-file configuration for ``anysearch explain``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequestModel(BaseModel):
    """Strict base model for explanation request documents."""

    model_config = ConfigDict(extra="forbid")


class RequestSource(RequestModel):
    """Exactly one explanation input source."""

    trace: str | None = None
    scenario: Path | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "RequestSource":
        """Reject absent or conflicting input sources."""

        if (self.trace is None) == (self.scenario is None):
            raise ValueError("source requires exactly one of trace or scenario")
        return self


class AttributionSettings(RequestModel):
    """Attribution and step-selection settings."""

    method: Literal["occlusion"] = "occlusion"
    focus: Literal["collisions", "all", "explicit"] = "collisions"
    max_steps: int = Field(default=50, ge=1)
    background: Literal["auto", "trace", "zeros", "mean"] = "auto"
    steps: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_steps(self) -> "AttributionSettings":
        """Keep explicit indices aligned with explicit focus."""

        if self.focus == "explicit" and not self.steps:
            raise ValueError("focus 'explicit' requires explanation.steps")
        if self.focus != "explicit" and self.steps:
            raise ValueError("explanation.steps is only valid with focus 'explicit'")
        return self


class OutputSettings(RequestModel):
    """Explanation artifact destination."""

    directory: Path | None = None


class ExplanationRequestFile(RequestModel):
    """Reusable explanation request loaded from YAML."""

    checkpoint: str = "latest"
    source: RequestSource
    explanation: AttributionSettings = Field(default_factory=AttributionSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    seed: int | None = None


def load_request_file(path: Path) -> ExplanationRequestFile:
    """Load and strictly validate an explanation request YAML."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"request {path} must contain a YAML mapping")
    return ExplanationRequestFile.model_validate(raw)
