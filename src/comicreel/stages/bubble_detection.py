"""Bubble-detection stage backends.

Returns a dict matching `BubbleDetectionResult` (see
`comicreel.artifacts.schemas`): page_id and a list of panels, each holding
the bubbles assigned to it, with pixel-space bboxes, an inferred type, an
(optional) tail point, and a speaker attribution from
`comicreel.speaker_attribution`.
"""

from __future__ import annotations

import math
from typing import Any

from comicreel.config import register_backend
from comicreel.geometry import BBox, assign_to_panel, center, to_bbox
from comicreel.pipeline.stage_base import Stage
from comicreel.speaker_attribution import attribute_speaker
from comicreel.stages.magi_model import run_magi_detection


@register_backend("bubble_detection", "magi")
class MagiBubbleDetector(Stage):
    """Bubble detection + speaker attribution using Magi v2.

    Makes its own call to the (shared, cached) Magi model rather than reusing
    `panel_detection`'s cached artifact, so this stage stays runnable on its
    own -- see stages/magi_model.py and the Phase 2 decisions log. Magi's own
    "characters" detections are used only internally, as attribution targets;
    they're raw per-page bboxes, not the persistent character_ids Phase 4
    assigns later.

    Bubble type is inferred, not detected: Magi doesn't classify bubble shape
    directly, so a text region with an associated tail is called "speech" and
    one without is called "narration" -- see the Phase 2 decisions log for
    why, and its limits (thought bubbles look like speech bubbles here).
    """

    name = "magi"

    def __init__(self, options: dict[str, Any] | None = None, device: str = "cpu"):
        self.options = options or {}
        self.device = device

    def run(self, comic_id: str, page_id: str, prior_artifacts: dict[str, Any]) -> dict[str, Any]:
        image_path = prior_artifacts["_input"]["image_path"]
        result = run_magi_detection(
            image_path,
            self.device,
            panel_detection_threshold=self.options.get("panel_detection_threshold", 0.2),
            character_detection_threshold=self.options.get("character_detection_threshold", 0.3),
            text_detection_threshold=self.options.get("text_detection_threshold", 0.3),
            tail_detection_threshold=self.options.get("tail_detection_threshold", 0.34),
            text_tail_matching_threshold=self.options.get("text_tail_matching_threshold", 0.3),
        )

        panel_boxes = [to_bbox(*p) for p in result["panels"]]
        text_boxes = [to_bbox(*t) for t in result["texts"]]
        tail_boxes = [to_bbox(*t) for t in result["tails"]]
        character_boxes = [to_bbox(*c) for c in result["characters"]]
        text_to_tail = dict(result["text_tail_associations"])

        # image diagonal, for attribution confidence decay -- derive from the
        # page's own detections rather than re-reading the image file.
        max_x = max((b[0] + b[2] for b in panel_boxes + text_boxes), default=1.0)
        max_y = max((b[1] + b[3] for b in panel_boxes + text_boxes), default=1.0)
        page_diagonal = math.hypot(max_x, max_y)

        # Characters grouped by panel: a bubble should only ever be attributed
        # to a character actually drawn in the *same* panel, never one from
        # elsewhere on the page.
        characters_by_panel: list[list[BBox]] = [[] for _ in panel_boxes]
        for char_box in character_boxes:
            if panel_boxes:
                characters_by_panel[assign_to_panel(char_box, panel_boxes)].append(char_box)

        max_angle_deg = self.options.get("attribution_max_angle_deg", 60.0)

        bubbles_by_panel: list[list[dict[str, Any]]] = [[] for _ in panel_boxes]
        for i, bbox in enumerate(text_boxes):
            tail_idx = text_to_tail.get(i)
            tail_point = center(tail_boxes[tail_idx]) if tail_idx is not None else None
            bubble_type = "speech" if tail_point is not None else "narration"

            panel_idx = assign_to_panel(bbox, panel_boxes) if panel_boxes else 0
            _px, _py, panel_w, panel_h = panel_boxes[panel_idx] if panel_boxes else (0, 0, 0, 0)
            panel_diagonal = math.hypot(panel_w, panel_h) or page_diagonal
            panel_characters = characters_by_panel[panel_idx] if panel_boxes else character_boxes

            char_idx, confidence = attribute_speaker(
                bbox, tail_point, panel_characters, panel_diagonal, max_angle_deg
            )
            attributed_bbox = panel_characters[char_idx] if char_idx is not None else None

            bubbles_by_panel[panel_idx].append(
                {
                    "bubble_id": f"{page_id}_b{i:03d}",
                    "bbox": bbox,
                    "type": bubble_type,
                    "tail_point": tail_point,
                    "attributed_character_bbox": attributed_bbox,
                    "attribution_confidence": confidence,
                }
            )

        panels = [
            {"panel_id": f"{page_id}_p{i:03d}", "bubbles": bubbles}
            for i, bubbles in enumerate(bubbles_by_panel)
        ]
        return {"page_id": page_id, "panels": panels}
