"""Panel-detection stage backends.

Both backends return a dict matching `PanelDetectionResult` (see
`comicreel.artifacts.schemas`): page_id, image_path, and a list of panels
with pixel-space bboxes and a `reading_order_index` assigned by
`comicreel.reading_order.order_panels`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comicreel.config import register_backend
from comicreel.pipeline.stage_base import Stage
from comicreel.reading_order import order_panels
from comicreel.stages.magi_model import run_magi_detection

BBox = tuple[float, float, float, float]


def _build_result(
    page_id: str, image_path: str, boxes: list[BBox], reading_direction: str
) -> dict[str, Any]:
    order = order_panels(boxes, reading_direction)
    rank_by_index = {panel_idx: rank for rank, panel_idx in enumerate(order)}
    panels = [
        {
            "panel_id": f"{page_id}_p{i:03d}",
            "bbox": boxes[i],
            "polygon": [],
            "reading_order_index": rank_by_index[i],
        }
        for i in range(len(boxes))
    ]
    panels.sort(key=lambda p: p["reading_order_index"])
    return {"page_id": page_id, "image_path": image_path, "panels": panels}


def _gutter_bands(is_bg_frac: np.ndarray, frac_threshold: float, min_gap: int) -> list[tuple[int, int]]:
    """Contiguous index ranges where `is_bg_frac` stays above `frac_threshold`,
    filtered to bands at least `min_gap` wide (thin gaps are noise, not gutters)."""
    idx = np.where(is_bg_frac > frac_threshold)[0]
    if len(idx) == 0:
        return []
    bands = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i > prev + 1:
            bands.append((start, prev))
            start = i
        prev = i
    bands.append((start, prev))
    return [b for b in bands if b[1] - b[0] + 1 >= min_gap]


def _split_on_gutters(
    lo: int, hi: int, is_bg_frac: np.ndarray, frac_threshold: float, min_gap: int, min_content: int
) -> list[tuple[int, int]]:
    """Cut [lo, hi) at the midpoints of any gutter bands found within it,
    dropping segments narrower than `min_content`."""
    bands = _gutter_bands(is_bg_frac[lo:hi], frac_threshold, min_gap)
    bands = [(lo + a, lo + b) for a, b in bands]
    cuts = [lo, *((a + b) // 2 for a, b in bands), hi]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= min_content]


def detect_panel_boxes_cv(
    image: np.ndarray,
    *,
    bg_brightness: int = 200,
    gutter_frac_threshold: float = 0.9,
    min_gutter_px: int = 5,
    min_panel_fraction: float = 0.06,
    max_bg_fraction_in_panel: float = 0.85,
) -> list[BBox]:
    """Find panel bounding boxes via recursive whitespace-gutter projection.

    Classic X-Y cut: a "gutter" is a full-width row (or, within a row, a
    full-height column) of mostly-background-brightness pixels. We split the
    page into row bands at horizontal gutters, then split each row band into
    panels at vertical gutters. This only needs the gutter/margin to be
    brighter than the art — unlike border-line detection, it tolerates
    broken, faded, or colored panel borders, which real scans have plenty of.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    is_bg = gray > bg_brightness

    min_h = max(1, int(min_panel_fraction * h))
    min_w = max(1, int(min_panel_fraction * w))

    row_frac = is_bg.mean(axis=1)
    row_segs = _split_on_gutters(0, h, row_frac, gutter_frac_threshold, min_gutter_px, min_h)

    boxes: list[BBox] = []
    for y0, y1 in row_segs:
        col_frac = is_bg[y0:y1, :].mean(axis=0)
        col_segs = _split_on_gutters(0, w, col_frac, gutter_frac_threshold, min_gutter_px, min_w)
        for x0, x1 in col_segs:
            if is_bg[y0:y1, x0:x1].mean() > max_bg_fraction_in_panel:
                continue  # this segment is itself mostly blank margin, not a panel
            boxes.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
    return boxes


def snap_boxes_to_gutters(
    image: np.ndarray,
    boxes: list[BBox],
    *,
    bg_brightness: int = 200,
    gutter_frac_threshold: float = 0.9,
    min_gutter_px: int = 8,
    max_search_fraction: float = 0.08,
) -> list[BBox]:
    """Expand each box outward, edge by edge, to the nearest genuine
    whitespace gutter, so a detector's box can't undershoot the real panel
    boundary and clip content (e.g. a caption strip near a panel's edge).

    A "genuine" gutter is a contiguous run of background-brightness rows/
    columns at least `min_gutter_px` wide -- long enough to rule out a
    narrow bright gap *inside* a caption box being mistaken for the page
    gutter (those are typically a few px, real gutters are wider). Only
    ever moves an edge outward from a confidently-found gutter; if none is
    found within `max_search_fraction` of the page, the edge is left as-is.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    is_bg = gray > bg_brightness
    search_y = max(1, int(max_search_fraction * h))
    search_x = max(1, int(max_search_fraction * w))

    snapped: list[BBox] = []
    for x, y, bw, bh in boxes:
        x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)

        lo = max(0, y0 - search_y)
        bands = _gutter_bands(is_bg[lo:y0, x0:x1].mean(axis=1), gutter_frac_threshold, min_gutter_px)
        new_y0 = lo + bands[-1][1] + 1 if bands else y0

        hi = min(h, y1 + search_y)
        bands = _gutter_bands(is_bg[y1:hi, x0:x1].mean(axis=1), gutter_frac_threshold, min_gutter_px)
        new_y1 = y1 + bands[0][0] if bands else y1

        lo = max(0, x0 - search_x)
        bands = _gutter_bands(
            is_bg[new_y0:new_y1, lo:x0].mean(axis=0), gutter_frac_threshold, min_gutter_px
        )
        new_x0 = lo + bands[-1][1] + 1 if bands else x0

        hi = min(w, x1 + search_x)
        bands = _gutter_bands(
            is_bg[new_y0:new_y1, x1:hi].mean(axis=0), gutter_frac_threshold, min_gutter_px
        )
        new_x1 = x1 + bands[0][0] if bands else x1

        if new_x1 <= new_x0 or new_y1 <= new_y0:
            snapped.append((x, y, bw, bh))  # degenerate snap result, keep original
        else:
            snapped.append((float(new_x0), float(new_y0), float(new_x1 - new_x0), float(new_y1 - new_y0)))
    return snapped


@register_backend("panel_detection", "cv_heuristic")
class CVHeuristicPanelDetector(Stage):
    """Zero-download, zero-GPU contour/gutter-based panel detector."""

    name = "cv_heuristic"

    def __init__(self, options: dict[str, Any] | None = None):
        self.options = options or {}

    def run(self, comic_id: str, page_id: str, prior_artifacts: dict[str, Any]) -> dict[str, Any]:
        image_path = prior_artifacts["_input"]["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        boxes = detect_panel_boxes_cv(
            image,
            bg_brightness=self.options.get("bg_brightness", 200),
            gutter_frac_threshold=self.options.get("gutter_frac_threshold", 0.9),
            min_gutter_px=self.options.get("min_gutter_px", 5),
            min_panel_fraction=self.options.get("min_panel_fraction", 0.06),
            max_bg_fraction_in_panel=self.options.get("max_bg_fraction_in_panel", 0.85),
        )
        reading_direction = self.options.get("reading_direction", "auto")
        return _build_result(page_id, image_path, boxes, reading_direction)


@register_backend("panel_detection", "magi")
class MagiPanelDetector(Stage):
    """Panel detection using the Magi v2 joint panel/bubble/character model.

    Only panel boxes are used at this phase; Magi's bubble/tail/character
    outputs are ignored here. `bubble_detection`'s `magi` backend makes its
    own call to the same (shared, cached) model rather than reusing this
    stage's result, so each stage stays independently runnable/cacheable --
    see stages/magi_model.py and the Phase 2 decisions log entry.
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
        )

        boxes: list[BBox] = []
        for x1, y1, x2, y2 in result["panels"]:
            boxes.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))

        if self.options.get("snap_to_gutters", True):
            page_image = cv2.imread(image_path)
            boxes = snap_boxes_to_gutters(
                page_image,
                boxes,
                bg_brightness=self.options.get("bg_brightness", 200),
                gutter_frac_threshold=self.options.get("gutter_frac_threshold", 0.9),
                min_gutter_px=self.options.get("snap_min_gutter_px", 8),
                max_search_fraction=self.options.get("snap_max_search_fraction", 0.08),
            )

        reading_direction = self.options.get("reading_direction", "auto")
        return _build_result(page_id, image_path, boxes, reading_direction)
