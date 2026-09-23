"""VLM-based dialogue extraction + per-panel action/setting/mood description.

Returns a dict matching `VLMExtractionResult` (see
`comicreel.artifacts.schemas`): page_id and a list of panels, each with a
VLM-generated action_description/setting/mood/onomatopoeia and the bubbles
in that panel with their OCR'd dialogue text.

Talks to a local Ollama server running the quantized `qwen2.5vl:7b` (~6GB).
An earlier version of this stage loaded the full, unquantized model directly
via `transformers` instead -- correct, but ~15GB to download and noticeably
slower to load/run on CPU/MPS than Ollama's optimized Metal/llama.cpp
backend. Removed after measuring the difference; see the Phase 3 decisions
log in progress.md if a non-Ollama fallback is ever needed again.

Makes its own Magi call (panels + texts), like `panel_detection` and
`bubble_detection` do, rather than depending on their cached artifacts, so
this stage stays independently runnable -- same pattern as Phase 1/2.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from comicreel.config import register_backend
from comicreel.geometry import assign_to_panel, to_bbox
from comicreel.pipeline.stage_base import Stage
from comicreel.stages.dialogue_ocr import build_bubble_extractions, ocr_page_bubbles
from comicreel.stages.magi_model import run_magi_detection
from comicreel.stages.panel_description_prompt import (
    PANEL_DESCRIPTION_PROMPT,
    PanelDescriptionResponse,
)

QWEN_OLLAMA_MODEL_ID = "qwen2.5vl:7b"


def _page_bubbles_and_panels(
    image_path: str, page_id: str, device: str, options: dict[str, Any]
) -> tuple[list[tuple[float, float, float, float]], list[list[dict[str, Any]]]]:
    """Magi panel/text detection, OCR, and assigning each OCR'd bubble to its
    panel. Returns (panel_boxes_x1y1x2y2, bubbles_by_panel)."""
    result = run_magi_detection(
        image_path,
        device,
        panel_detection_threshold=options.get("panel_detection_threshold", 0.2),
        text_detection_threshold=options.get("text_detection_threshold", 0.3),
    )

    panel_boxes_x1y1x2y2 = result["panels"]
    panel_boxes = [to_bbox(*p) for p in panel_boxes_x1y1x2y2]
    text_boxes = result["texts"]  # Magi's own [x1,y1,x2,y2] format, needed as-is for OCR
    text_bboxes = [to_bbox(*t) for t in text_boxes]

    ocr_results = ocr_page_bubbles(image_path, text_boxes, device)
    bubble_extractions = build_bubble_extractions(page_id, ocr_results)

    bubbles_by_panel: list[list[dict[str, Any]]] = [[] for _ in panel_boxes]
    for bbox, extraction in zip(text_bboxes, bubble_extractions, strict=True):
        if panel_boxes:
            bubbles_by_panel[assign_to_panel(bbox, panel_boxes)].append(extraction)

    return panel_boxes_x1y1x2y2, bubbles_by_panel


def describe_panel_with_ollama(
    image: Image.Image, model: str = QWEN_OLLAMA_MODEL_ID
) -> dict[str, Any]:
    """Run the structured panel-description prompt via a local Ollama
    server. Uses Ollama's native JSON-schema-constrained output (the
    `format` param) rather than `parse_panel_description_response`'s
    tolerant text parsing, since Ollama guarantees schema-valid JSON."""
    import io

    import ollama

    buf = io.BytesIO()
    image.save(buf, format="PNG")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PANEL_DESCRIPTION_PROMPT,
                "images": [buf.getvalue()],
            }
        ],
        format=PanelDescriptionResponse.model_json_schema(),
        options={"temperature": 0},
    )
    parsed = PanelDescriptionResponse.model_validate_json(response["message"]["content"])
    return {
        "action_description": parsed.action_description.strip(),
        "setting": parsed.setting.strip(),
        "mood": parsed.mood.strip(),
        "onomatopoeia": [o.strip() for o in parsed.onomatopoeia if o.strip()],
    }


@register_backend("vlm_extraction", "qwen_vl_ollama")
class QwenVLOllamaExtractor(Stage):
    name = "qwen_vl_ollama"

    def __init__(self, options: dict[str, Any] | None = None, device: str = "cpu"):
        self.options = options or {}
        self.device = device  # only used for the Magi/OCR half; Ollama manages its own device

    def run(self, comic_id: str, page_id: str, prior_artifacts: dict[str, Any]) -> dict[str, Any]:
        image_path = prior_artifacts["_input"]["image_path"]
        panel_boxes_x1y1x2y2, bubbles_by_panel = _page_bubbles_and_panels(
            image_path, page_id, self.device, self.options
        )

        page_image = Image.open(image_path).convert("RGB")
        model = self.options.get("ollama_model", QWEN_OLLAMA_MODEL_ID)

        panels: list[dict[str, Any]] = []
        for i, (x1, y1, x2, y2) in enumerate(panel_boxes_x1y1x2y2):
            crop = page_image.crop((int(x1), int(y1), int(x2), int(y2)))
            description = describe_panel_with_ollama(crop, model)
            panels.append(
                {
                    "panel_id": f"{page_id}_p{i:03d}",
                    "action_description": description["action_description"],
                    "setting": description["setting"],
                    "mood": description["mood"],
                    "onomatopoeia": description["onomatopoeia"],
                    "bubbles": bubbles_by_panel[i],
                }
            )

        return {"page_id": page_id, "panels": panels}
