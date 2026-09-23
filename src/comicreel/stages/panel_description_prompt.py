"""The structured per-panel prompt sent to whichever VLM backend is chosen
for Phase 3, and a parser for its response.

Kept separate from any specific VLM backend so the prompt (and its parsing)
can be written, tested, and iterated on before a model is even downloaded --
every VLM backend that gets added later reuses the same prompt and parser.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class PanelDescriptionResponse(BaseModel):
    """Schema for backends (like Ollama) that support constrained/guaranteed
    structured JSON output directly, instead of needing
    `parse_panel_description_response`'s tolerant text parsing."""

    action_description: str
    setting: str
    mood: str
    onomatopoeia: list[str] = Field(default_factory=list)


PANEL_DESCRIPTION_PROMPT = """You are analyzing a single panel from a comic book. Look at the image and respond with ONLY a JSON object in exactly this format, no other text:

{
  "action_description": "one or two sentences describing what is physically happening in this panel",
  "setting": "a short phrase describing the location/environment (e.g. 'a city street', 'a bedroom at night')",
  "mood": "one or two words for the emotional tone of the panel (e.g. 'tense', 'comedic', 'melancholic')",
  "onomatopoeia": ["list", "of", "any", "sound-effect", "lettering", "visible", "in", "the", "panel", "e.g. BANG, WHOOSH"]
}

If there is no onomatopoeia visible, use an empty list. Do not describe the dialogue text in speech bubbles -- focus only on the visual action, setting, and mood."""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_REQUIRED_STR_FIELDS = ("action_description", "setting", "mood")


def parse_panel_description_response(raw_text: str) -> dict[str, Any]:
    """Extract the structured fields from a VLM's raw text response.

    Tolerant of models that wrap the JSON in a markdown code fence or add
    stray prose before/after it -- pulls out the first `{...}` block found.
    Raises ValueError if no valid, complete JSON object can be recovered.
    """
    match = _JSON_OBJECT_RE.search(raw_text)
    if not match:
        raise ValueError(f"No JSON object found in VLM response: {raw_text!r}")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in VLM response: {raw_text!r}") from exc

    if not isinstance(parsed, dict):
        # ValueError (not TypeError), deliberately: callers should be able to
        # catch one exception type for "this VLM response couldn't be used."
        raise ValueError(  # noqa: TRY004
            f"Expected a JSON object, got {type(parsed).__name__}: {raw_text!r}"
        )

    missing = [f for f in _REQUIRED_STR_FIELDS if f not in parsed]
    if missing:
        raise ValueError(f"VLM response missing required field(s) {missing}: {raw_text!r}")

    onomatopoeia = parsed.get("onomatopoeia", [])
    if not isinstance(onomatopoeia, list):
        onomatopoeia = []

    return {
        "action_description": str(parsed["action_description"]).strip(),
        "setting": str(parsed["setting"]).strip(),
        "mood": str(parsed["mood"]).strip(),
        "onomatopoeia": [str(o).strip() for o in onomatopoeia if str(o).strip()],
    }
