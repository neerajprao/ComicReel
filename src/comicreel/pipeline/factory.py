"""Builds a Stage instance for a pipeline stage from its resolved config."""

from __future__ import annotations

import inspect

from comicreel.config import PipelineConfig, get_backend_class
from comicreel.pipeline.stage_base import Stage


def build_stage(stage_name: str, config: PipelineConfig) -> Stage:
    stage_cfg = config.stages[stage_name]
    stage_cls = get_backend_class(stage_name, stage_cfg.backend)

    kwargs: dict = {"options": stage_cfg.options}
    if "device" in inspect.signature(stage_cls.__init__).parameters:
        kwargs["device"] = config.resolve_device(stage_name)

    return stage_cls(**kwargs)
