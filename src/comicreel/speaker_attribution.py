"""Attribute a dialogue bubble to the character bbox that most likely said it.

Two strategies, tried in order:
1. Tail-pointing: a speech bubble's tail points at its speaker. Given the
   bubble and the point where its tail tip is, find the character bbox that
   lies closest to the tail's pointing direction.
2. Nearest-character fallback: when there's no tail (e.g. narration boxes)
   or no character lines up with the tail direction, attribute to whichever
   detected character bbox on the page is spatially closest to the bubble.

Works on raw per-page character bboxes (not the persistent character_ids
Phase 4 assigns via cross-page clustering) -- see the Phase 2 decisions log
in progress.md for why that's the right layering.
"""

from __future__ import annotations

import math

BBox = tuple[float, float, float, float]  # x, y, w, h
Point = tuple[float, float]


def _center(box: BBox) -> Point:
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_between(u: Point, v: Point) -> float:
    """Angle in degrees between two 2D vectors, in [0, 180]."""
    mag_u, mag_v = math.hypot(*u), math.hypot(*v)
    if mag_u == 0 or mag_v == 0:
        return 180.0
    cos_theta = (u[0] * v[0] + u[1] * v[1]) / (mag_u * mag_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # guard float drift past [-1, 1]
    return math.degrees(math.acos(cos_theta))


def tail_pointing_attribution(
    bubble_bbox: BBox,
    tail_point: Point,
    character_bboxes: list[BBox],
    max_angle_deg: float = 60.0,
) -> tuple[int | None, float]:
    """Find the character index whose direction from the tail tip best
    matches the bubble-center-to-tail-tip pointing direction.

    Returns (index, confidence) or (None, 0.0) if nothing lines up within
    `max_angle_deg`. Confidence decays linearly from 1.0 (dead-on) to 0.0
    (at the angle threshold).
    """
    if not character_bboxes:
        return None, 0.0

    bubble_center = _center(bubble_bbox)
    pointing_dir = (tail_point[0] - bubble_center[0], tail_point[1] - bubble_center[1])

    best_idx, best_angle = None, max_angle_deg
    for i, char_box in enumerate(character_bboxes):
        char_center = _center(char_box)
        to_char = (char_center[0] - tail_point[0], char_center[1] - tail_point[1])
        angle = _angle_between(pointing_dir, to_char)
        if angle <= best_angle:
            best_idx, best_angle = i, angle

    if best_idx is None:
        return None, 0.0
    confidence = max(0.0, 1.0 - best_angle / max_angle_deg)
    return best_idx, confidence


def nearest_character_attribution(
    bubble_bbox: BBox,
    character_bboxes: list[BBox],
    page_diagonal: float,
) -> tuple[int | None, float]:
    """Attribute to the closest character bbox by center distance.
    Confidence decays linearly to 0 at a full page-diagonal away."""
    if not character_bboxes or page_diagonal <= 0:
        return None, 0.0

    bubble_center = _center(bubble_bbox)
    distances = [_distance(bubble_center, _center(c)) for c in character_bboxes]
    best_idx = min(range(len(distances)), key=lambda i: distances[i])
    confidence = max(0.0, 1.0 - distances[best_idx] / page_diagonal)
    return best_idx, confidence


def attribute_speaker(
    bubble_bbox: BBox,
    tail_point: Point | None,
    character_bboxes: list[BBox],
    page_diagonal: float,
    max_angle_deg: float = 60.0,
) -> tuple[int | None, float]:
    """Try tail-pointing first (when a tail exists), fall back to nearest-character."""
    if tail_point is not None:
        idx, confidence = tail_pointing_attribution(
            bubble_bbox, tail_point, character_bboxes, max_angle_deg
        )
        if idx is not None:
            return idx, confidence
    return nearest_character_attribution(bubble_bbox, character_bboxes, page_diagonal)
