"""Pipeline configuration and the per-stage backend registry.

Each model-using stage (panel detector, bubble detector, VLM, ...) has one or
more implementations registered under its stage name via `@register_backend`.
`configs/pipeline.yaml` then picks which implementation + device to use per
stage, so the same codebase runs on anything from a CPU-only laptop to a
rented GPU without code changes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from comicreel.utils.device import resolve_device

Device = Literal["auto", "cpu", "cuda", "mps"]

_REGISTRY: dict[str, dict[str, type]] = {}


def register_backend(stage: str, name: str) -> Callable[[type], type]:
    """Class decorator that registers a backend implementation under a stage name."""

    def decorator(cls: type) -> type:
        _REGISTRY.setdefault(stage, {})[name] = cls
        return cls

    return decorator


def get_backend_class(stage: str, name: str) -> type:
    try:
        return _REGISTRY[stage][name]
    except KeyError as exc:
        available = sorted(_REGISTRY.get(stage, {}))
        raise ValueError(
            f"Unknown backend '{name}' for stage '{stage}'. Available: {available}"
        ) from exc


def registered_backends(stage: str) -> list[str]:
    return sorted(_REGISTRY.get(stage, {}))


class StageConfig(BaseModel):
    backend: str
    device: Device = "auto"
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    stages: dict[str, StageConfig]

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(stages={k: StageConfig(**v) for k, v in raw.get("stages", {}).items()})

    def resolve_device(self, stage: str) -> str:
        return resolve_device(self.stages[stage].device)

    def validate_backends(self) -> None:
        """Raise if any configured backend isn't registered (skips stages with no
        implementations registered yet, since those are added phase by phase)."""
        for stage, cfg in self.stages.items():
            available = registered_backends(stage)
            if available and cfg.backend not in available:
                raise ValueError(
                    f"Backend '{cfg.backend}' not registered for stage '{stage}'. "
                    f"Available: {available}"
                )
