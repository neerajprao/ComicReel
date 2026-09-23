"""Color-histogram similarity between character crops.

A Phase 4 measurement found DINOv2/CLIP alone confuse characters even in a
visually distinct, brightly-colored modern comic (Archie) -- not just the
muddy-palette 1940s funny-animal comic this project started with. In both
cases, a human actually tells these flat-colored cartoon characters apart
mostly by costume/hair color (Archie's orange hair, Chuck's red coat,
Weatherbee's blue suit), a signal general-purpose photo-trained embedders
don't weight the way we'd want. This module is that missing signal: a hue
histogram of each crop's saturated, non-background pixels, meant to be fused
with the embedding vector (see `stages/character_embedding.py`) rather than
replace it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_BINS = 16
# Pixels outside these bounds are panel linework (near-black), paper/highlight
# background (near-white), or too washed-out to carry real costume-color
# identity signal -- excluded so they don't dilute the histogram.
MIN_SATURATION = 60
MIN_VALUE = 40
MAX_VALUE = 250


def color_histogram(image: Image.Image, bins: int = DEFAULT_BINS) -> list[float]:
    """Normalized hue histogram (sums to 1) of `image`'s saturated,
    mid-brightness pixels. All-zero (never negative-sum) if no pixel
    qualifies, e.g. a fully grayscale crop."""
    hsv = np.array(image.convert("HSV"))
    hue, saturation, value = (hsv[..., i].astype(float) for i in range(3))

    mask = (saturation > MIN_SATURATION) & (value > MIN_VALUE) & (value < MAX_VALUE)
    hues = hue[mask]
    if hues.size == 0:
        return [0.0] * bins

    hist, _ = np.histogram(hues, bins=bins, range=(0, 255))
    total = hist.sum()
    return (hist / total).tolist() if total > 0 else [0.0] * bins


def histogram_similarity(hist_a: list[float], hist_b: list[float]) -> float:
    """Histogram intersection: the fraction of the two color distributions
    that overlaps, in [0, 1]. 1.0 for identical distributions, 0.0 for
    disjoint ones (e.g. all-orange vs all-blue)."""
    a, b = np.asarray(hist_a, dtype=float), np.asarray(hist_b, dtype=float)
    return float(np.minimum(a, b).sum())
