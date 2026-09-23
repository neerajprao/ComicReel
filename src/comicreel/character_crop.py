"""Crop a character detection box down to just its head/face region.

Magi has no dedicated face/head detector, only whole-character boxes (which
include clothing, pose, and background) -- and a Phase 4 measurement found
that embedding the *whole* character box gives CLIP/DINOv2 too little
identity signal to separate on: outfit, pose, and background dominate the
embedding over which character it actually is. Taking just the top portion
of the box (where the head reliably is, for a standing/sitting figure) and
narrowing the width (to exclude shoulders/arms) is a cheap, zero-download
heuristic to concentrate the crop on the face -- same "simple geometric rule
over the detector's raw box" pattern as `snap_boxes_to_gutters` in
`stages/panel_detection.py`.
"""

from __future__ import annotations

BBox = tuple[float, float, float, float]  # x, y, w, h


def head_crop_bbox(
    bbox: BBox, *, top_fraction: float = 0.45, width_fraction: float = 0.7
) -> BBox:
    """Return the top `top_fraction` of `bbox`'s height, horizontally
    narrowed to the centered `width_fraction` of its width."""
    x, y, w, h = bbox
    head_h = h * top_fraction
    head_w = w * width_fraction
    head_x = x + (w - head_w) / 2
    return (head_x, y, head_w, head_h)
