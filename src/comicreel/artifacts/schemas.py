"""Pydantic schemas for every inter-stage JSON artifact in the ComicReel pipeline.

Each stage reads the prior stage's artifact(s) and writes its own, keyed by
page_id -> panel_id -> bubble_id/character_id. These schemas are the contract
between stages and what `StageCache` validates against.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # x, y, w, h


# ---- Stage 1: panel detection ----


class Panel(BaseModel):
    panel_id: str
    bbox: BBox
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    reading_order_index: int


class PanelDetectionResult(BaseModel):
    page_id: str
    image_path: str
    panels: list[Panel]


# ---- Stage 2: bubble detection & speaker attribution ----

BubbleType = Literal["speech", "thought", "narration", "sfx"]


class Bubble(BaseModel):
    bubble_id: str
    bbox: BBox
    type: BubbleType
    tail_point: tuple[float, float] | None = None
    attributed_character_bbox: BBox | None = None
    attribution_confidence: float = 0.0


class PanelBubbles(BaseModel):
    panel_id: str
    bubbles: list[Bubble]


class BubbleDetectionResult(BaseModel):
    page_id: str
    panels: list[PanelBubbles]


# ---- Stage 3: VLM dialogue + action extraction ----


class BubbleExtraction(BaseModel):
    bubble_id: str
    dialogue_text: str
    ocr_confidence: float = 0.0


class PanelExtraction(BaseModel):
    panel_id: str
    action_description: str
    setting: str = ""
    mood: str = ""
    onomatopoeia: list[str] = Field(default_factory=list)
    bubbles: list[BubbleExtraction] = Field(default_factory=list)


class VLMExtractionResult(BaseModel):
    page_id: str
    panels: list[PanelExtraction]


# ---- Stage 4: character identity tracking ----


class Character(BaseModel):
    character_id: str
    representative_crop: str
    name: str | None = None


class CharacterAssignment(BaseModel):
    panel_id: str
    bubble_id: str | None = None
    character_id: str
    similarity: float = 0.0


class CharacterIndex(BaseModel):
    characters: list[Character]
    panel_assignments: list[CharacterAssignment]


# ---- Stage 5: script generation (LLM, emotion-tagged) ----

Emotion = Literal[
    "neutral", "anger", "fear", "sadness", "joy", "surprise", "disgust", "whisper", "shout",
]
LineType = Literal["dialogue", "narration", "thought"]


class ScriptLine(BaseModel):
    panel_id: str
    bubble_id: str | None = None
    character_id: str | None = None
    type: LineType
    text: str
    emotion: Emotion = "neutral"
    intensity: int = Field(default=2, ge=1, le=5)


class ScriptResult(BaseModel):
    page_id: str
    script_lines: list[ScriptLine]


# ---- Stage 6: emotion-controllable TTS ----


class AudioLine(BaseModel):
    panel_id: str
    bubble_id: str | None = None
    character_id: str | None = None
    audio_path: str
    duration_sec: float


class AudioManifest(BaseModel):
    page_id: str
    lines: list[AudioLine]


# ---- Stage 7: Tier 3 animation (image-to-video) ----


class Clip(BaseModel):
    panel_id: str
    clip_path: str
    duration_sec: float
    motion_prompt: str


class ClipManifest(BaseModel):
    page_id: str
    clips: list[Clip]


# ---- Stage 8: mood music & sound effects ----


class MusicTrack(BaseModel):
    scene_id: str
    pages: list[str]
    path: str


class SFXCue(BaseModel):
    panel_id: str
    onomatopoeia: str
    path: str


class MusicSFXManifest(BaseModel):
    music_tracks: list[MusicTrack]
    sfx: list[SFXCue]


# ---- Stage 9: compositing & export ----


class TimelineEntry(BaseModel):
    panel_id: str
    start_sec: float
    end_sec: float
    clip_path: str | None = None
    dialogue_audio_paths: list[str] = Field(default_factory=list)
    music_path: str | None = None
    sfx_paths: list[str] = Field(default_factory=list)


class Timeline(BaseModel):
    comic_id: str
    entries: list[TimelineEntry]
    output_path: str
