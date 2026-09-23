"""OCR for bubble dialogue text.

Two independent sources per bubble, both zero-extra-download:
1. Magi's built-in TrOCR (microsoft/trocr-base-printed) -- already part of
   the Magi weights we have from Phase 1/2. English-only, matching our
   English sample pages.
2. Tesseract (via pytesseract) -- already installed as a system binary.

Neither one is reliably better than the other on real 1940s comic lettering
(measured on our own sample pages: Tesseract is often more complete on
clear, larger text, but occasionally fails outright -- empty output, or
truncated to a couple of words despite reporting high confidence -- in a way
Magi's OCR usually doesn't). So rather than a strict primary/fallback, we run
both and keep whichever recovered more actual text, on the reasoning that a
longer (if imperfect) transcription beats a short, confidently wrong one.
Confidence-based selection alone doesn't work here since Tesseract's
per-word confidence isn't a reliable signal for *completeness*.

Magi's `predict_ocr` doesn't return a confidence score at all, so its
results get a fixed placeholder confidence (see MAGI_OCR_CONFIDENCE below)
-- documented here rather than faked as something more precise than it is.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from comicreel.stages.magi_model import get_magi_model

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 (Magi's own format)

MAGI_OCR_CONFIDENCE = 0.85


def ocr_texts_with_magi(image_array: np.ndarray, text_boxes: list[BBox], device: str) -> list[str]:
    """Batch-OCR every text box on one page in a single Magi call.
    `text_boxes` must be in Magi's own [x1, y1, x2, y2] format (i.e. the raw
    `texts` list from predict_detections_and_associations, not our (x,y,w,h)
    schema bboxes)."""
    if not text_boxes:
        return []
    import torch

    model = get_magi_model(device)
    with torch.no_grad():
        [texts] = model.predict_ocr([image_array], [text_boxes])
    return [t.strip() for t in texts]


def ocr_text_with_tesseract(image_array: np.ndarray, box_x1y1x2y2: BBox) -> tuple[str, float]:
    """OCR a single crop with Tesseract. Returns (text, avg_word_confidence in [0,1])."""
    import pytesseract
    from pytesseract import Output

    x1, y1, x2, y2 = (int(v) for v in box_x1y1x2y2)
    crop = Image.fromarray(image_array[max(0, y1) : y2, max(0, x1) : x2])
    if crop.width == 0 or crop.height == 0:
        return "", 0.0

    data = pytesseract.image_to_data(crop, output_type=Output.DICT)
    words = [w.strip() for w in data["text"] if w.strip()]
    confidences = [float(c) for c, w in zip(data["conf"], data["text"], strict=True) if w.strip() and float(c) >= 0]

    text = " ".join(words)
    confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return text, confidence


def _alpha_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def pick_better_ocr_result(
    magi_text: str, tesseract_result: tuple[str, float]
) -> tuple[str, float]:
    """Choose whichever OCR result recovered more actual text (by letter
    count), on the theory that a longer, imperfect transcription beats a
    short, confidently-wrong one -- see the module docstring for the real
    comparison data this is based on."""
    tesseract_text, tesseract_confidence = tesseract_result
    if _alpha_count(tesseract_text) > _alpha_count(magi_text):
        return tesseract_text, tesseract_confidence
    return magi_text, MAGI_OCR_CONFIDENCE


def ocr_page_bubbles(
    image_path: str, text_boxes: list[BBox], device: str
) -> list[tuple[str, float]]:
    """OCR every bubble/text region on a page with both Magi and Tesseract,
    keeping the better result per box (see `pick_better_ocr_result`).
    Returns (text, confidence) per box, same order as `text_boxes`."""
    image_array = np.array(Image.open(image_path).convert("RGB"))
    magi_texts = ocr_texts_with_magi(image_array, text_boxes, device)

    results: list[tuple[str, float]] = []
    for box, magi_text in zip(text_boxes, magi_texts, strict=True):
        tesseract_result = ocr_text_with_tesseract(image_array, box)
        results.append(pick_better_ocr_result(magi_text, tesseract_result))
    return results


def build_bubble_extractions(
    page_id: str, ocr_results: list[tuple[str, float]]
) -> list[dict]:
    """Turn per-box OCR results into `BubbleExtraction`-shaped dicts, using
    the same `f"{page_id}_b{i:03d}"` bubble_id scheme as
    `bubble_detection.py`, so IDs line up across stages (both stages derive
    their own boxes from independent Magi calls on the same image, which is
    deterministic -- see the Phase 2 decisions log)."""
    return [
        {
            "bubble_id": f"{page_id}_b{i:03d}",
            "dialogue_text": text,
            "ocr_confidence": confidence,
        }
        for i, (text, confidence) in enumerate(ocr_results)
    ]
