"""Shared, cached CLIP / DINOv2 embedding models for character identity
tracking.

Both follow the same "load once per (backend, model_id, device), cache by
that key" pattern as `stages/magi_model.py`, so re-running
`character_embedding` across a comic's pages in one process doesn't reload
weights each time. Unlike Magi, these are small enough (one model each) that
no separate device-agnostic sharing module is needed -- both live here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

_MODELS: dict[tuple[str, str, str], Any] = {}

DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_DINOV2_MODEL_ID = "facebook/dinov2-base"


def _get_clip(model_id: str, device: str) -> tuple[Any, Any]:
    key = ("clip", model_id, device)
    if key not in _MODELS:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
        _MODELS[key] = (model, processor)
    return _MODELS[key]


def embed_image_clip(
    image: Image.Image, *, device: str = "cpu", model_id: str = DEFAULT_CLIP_MODEL_ID
) -> np.ndarray:
    import torch

    model, processor = _get_clip(model_id, device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features[0].cpu().numpy()


def _get_dinov2(model_id: str, device: str) -> tuple[Any, Any]:
    key = ("dinov2", model_id, device)
    if key not in _MODELS:
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained(model_id).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(model_id)
        _MODELS[key] = (model, processor)
    return _MODELS[key]


def embed_image_dinov2(
    image: Image.Image, *, device: str = "cpu", model_id: str = DEFAULT_DINOV2_MODEL_ID
) -> np.ndarray:
    import torch

    model, processor = _get_dinov2(model_id, device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].cpu().numpy()
