"""Small bbox geometry helpers shared across stages.

`BBox` here is always (x, y, w, h) -- our schema's pixel-space format.
Magi's own outputs are (x1, y1, x2, y2); convert with `to_bbox` first.
"""

from __future__ import annotations

import math

BBox = tuple[float, float, float, float]  # x, y, w, h


def center(box: BBox) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def to_bbox(x1: float, y1: float, x2: float, y2: float) -> BBox:
    """Convert Magi's (x1, y1, x2, y2) format to our (x, y, w, h) schema format."""
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def contains(box: BBox, point: tuple[float, float]) -> bool:
    x, y, w, h = box
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def assign_to_panel(item_box: BBox, panel_boxes: list[BBox]) -> int:
    """Index of the panel whose bbox contains `item_box`'s center; if none
    does (e.g. an item straddling a panel edge), the panel with the closest
    center instead."""
    item_center = center(item_box)
    for i, panel_box in enumerate(panel_boxes):
        if contains(panel_box, item_center):
            return i
    distances = [math.dist(item_center, center(p)) for p in panel_boxes]
    return min(range(len(distances)), key=lambda i: distances[i])
