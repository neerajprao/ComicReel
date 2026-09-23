"""Spatial reading-order assignment for detected panels.

Groups panel bounding boxes into rows by vertical (y-axis) overlap, then
orders rows top-to-bottom and panels within each row left-to-right (ltr) or
right-to-left (rtl). This is the project's own general-purpose ordering pass
so the same logic applies no matter which detector backend (CV heuristic,
Magi, YOLO, ...) produced the panel boxes.
"""

from __future__ import annotations

from typing import Literal

BBox = tuple[float, float, float, float]  # x, y, w, h
Direction = Literal["auto", "ltr", "rtl"]


def _y_overlap_fraction(a: BBox, b: BBox) -> float:
    ay0, ay1 = a[1], a[1] + a[3]
    by0, by1 = b[1], b[1] + b[3]
    overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
    shorter = min(ay1 - ay0, by1 - by0)
    return overlap / shorter if shorter > 0 else 0.0


def group_into_rows(bboxes: list[BBox], overlap_threshold: float = 0.3) -> list[list[int]]:
    """Cluster panel indices into reading-order rows by vertical overlap.

    Panels are visited in order of vertical center. A panel joins the first
    existing row it vertically overlaps (by at least `overlap_threshold` of
    the shorter panel's height); otherwise it starts a new row. Rows are then
    sorted top-to-bottom by their topmost panel.
    """
    visit_order = sorted(range(len(bboxes)), key=lambda i: bboxes[i][1] + bboxes[i][3] / 2)
    rows: list[list[int]] = []
    for i in visit_order:
        for row in rows:
            if any(_y_overlap_fraction(bboxes[i], bboxes[j]) >= overlap_threshold for j in row):
                row.append(i)
                break
        else:
            rows.append([i])
    rows.sort(key=lambda row: min(bboxes[i][1] for i in row))
    return rows


def order_panels(bboxes: list[BBox], direction: Direction = "auto") -> list[int]:
    """Return panel indices (into `bboxes`) in reading order.

    `direction` picks left-to-right or right-to-left ordering within a row.
    "auto" currently resolves to left-to-right (western comics), since
    automatic manga-vs-western format detection isn't implemented yet.
    """
    if not bboxes:
        return []
    ltr = direction != "rtl"
    reading_order: list[int] = []
    for row in group_into_rows(bboxes):
        reading_order.extend(sorted(row, key=lambda i: bboxes[i][0], reverse=not ltr))
    return reading_order
