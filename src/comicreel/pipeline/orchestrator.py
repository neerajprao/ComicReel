"""Runs pipeline stages in order for a page, with caching and resume support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comicreel.artifacts.cache import StageCache
from comicreel.config import PipelineConfig
from comicreel.pipeline.stage_base import Stage

STAGE_ORDER = [
    "panel_detection",
    "bubble_detection",
    "vlm_extraction",
    "character_embedding",
    "script_generation",
    "tts",
    "animation",
    "music_sfx",
    "compositing",
]


class Orchestrator:
    def __init__(self, config: PipelineConfig, cache_root: Path | str, stages: dict[str, Stage]):
        self.config = config
        self.cache = StageCache(cache_root)
        self.stages = stages

    def run_page(
        self,
        comic_id: str,
        page_id: str,
        image_path: Path | str,
        from_stage: str | None = None,
        to_stage: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        start = STAGE_ORDER.index(from_stage) if from_stage else 0
        end = STAGE_ORDER.index(to_stage) if to_stage else len(STAGE_ORDER) - 1

        # "_input" is a reserved pseudo-stage: raw per-page inputs that stage 1
        # (panel_detection) needs but that no prior stage produces.
        artifacts: dict[str, Any] = {"_input": {"image_path": str(image_path)}}
        for stage_name in STAGE_ORDER[start : end + 1]:
            stage = self.stages.get(stage_name)
            if stage is None:
                raise ValueError(f"No stage implementation registered for '{stage_name}'")

            stage_cfg = self.config.stages[stage_name]
            input_hash = self.cache.hash_inputs(config=stage_cfg.model_dump())

            if not force and self.cache.has(comic_id, page_id, stage_name, input_hash):
                artifacts[stage_name] = self.cache.get(comic_id, page_id, stage_name, input_hash)
                continue

            result = stage.run(comic_id, page_id, artifacts)
            self.cache.set(comic_id, page_id, stage_name, input_hash, result)
            artifacts[stage_name] = result

        return artifacts
