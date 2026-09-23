"""Shared, process-wide cache for the Magi v2 model.

Both `panel_detection`'s and `bubble_detection`'s Magi backends need the same
~2GB model. Without sharing, running both stages in one process would load it
twice. Keyed by device, since a model moved to one device isn't usable as-is
on another.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MODELS: dict[str, Any] = {}


def get_magi_model(device: str) -> Any:
    if device not in _MODELS:
        from transformers import AutoModel

        model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
        _MODELS[device] = model.to(device).eval()
    return _MODELS[device]


def run_magi_detection(image_path: str, device: str, **detection_kwargs: Any) -> dict[str, Any]:
    """Run Magi's per-page joint detection (panels/texts/tails/characters +
    their associations) on one image. Each call is a fresh forward pass --
    the model is shared and cached, but not its per-image outputs, since
    different stages pass different detection thresholds."""
    import torch
    from PIL import Image

    model = get_magi_model(device)
    image_array = np.array(Image.open(image_path).convert("RGB"))

    with torch.no_grad():
        [result] = model.predict_detections_and_associations([image_array], **detection_kwargs)
    return result
