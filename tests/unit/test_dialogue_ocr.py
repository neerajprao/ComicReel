import numpy as np
from PIL import Image, ImageDraw, ImageFont

from comicreel.artifacts.schemas import BubbleExtraction
from comicreel.stages.dialogue_ocr import (
    MAGI_OCR_CONFIDENCE,
    build_bubble_extractions,
    ocr_text_with_tesseract,
    pick_better_ocr_result,
)

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _render_text(text: str, size=(300, 100)) -> np.ndarray:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, 40)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 20), text, fill="black", font=font)
    return np.array(img)


def test_reads_simple_rendered_text():
    image = _render_text("HELLO")
    text, confidence = ocr_text_with_tesseract(image, (0, 0, 300, 100))
    assert text == "HELLO"
    assert confidence > 0.5


def test_empty_box_returns_empty_result():
    image = _render_text("HELLO")
    text, confidence = ocr_text_with_tesseract(image, (10, 10, 10, 10))
    assert text == ""
    assert confidence == 0.0


def test_blank_crop_has_no_confident_text():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    text, confidence = ocr_text_with_tesseract(image, (0, 0, 300, 100))
    assert text == ""
    assert confidence == 0.0


# pick_better_ocr_result cases below are taken directly from a real
# side-by-side comparison run on tests/fixtures/sample_pages/page_04.jpg
# (see progress.md's Phase 3 decisions log) -- both engines' real failure
# modes on real 1940s comic lettering, not synthetic edge cases.


def test_prefers_tesseract_when_it_recovers_more_text():
    # Magi mangled "MOON BLOT OUT" into "monbotout"; Tesseract got it almost
    # verbatim.
    magi_text = "get them here, 'going-free of charge!' and watch the monbotout the sun!"
    tesseract_result = ("GET THEM HERE, FOLKS--FREE OF CHARGE! AND WATCH THE MOON BLOT OUT THE SUN?", 0.77)
    text, confidence = pick_better_ocr_result(magi_text, tesseract_result)
    assert text == tesseract_result[0]
    assert confidence == 0.77


def test_prefers_magi_when_tesseract_fails_outright():
    # Tesseract returned nothing at all for this box.
    magi_text = "Well,"
    text, confidence = pick_better_ocr_result(magi_text, ("", 0.0))
    assert text == "Well,"
    assert confidence == MAGI_OCR_CONFIDENCE


def test_prefers_magi_when_tesseract_truncates_badly_despite_high_confidence():
    # Real case: Tesseract reported 0.85 confidence for a 3-letter fragment
    # ("FOR") of a full sentence Magi recovered mostly intact -- proves
    # confidence alone isn't a safe signal for completeness.
    magi_text = "nothing else left (but for us to take,count)."
    text, confidence = pick_better_ocr_result(magi_text, ("FOR", 0.85))
    assert text == magi_text
    assert confidence == MAGI_OCR_CONFIDENCE


def test_prefers_magi_when_tesseract_produces_garbage():
    magi_text = "Thank you!"
    text, confidence = pick_better_ocr_result(magi_text, ("ue Es", 0.39))
    assert text == "Thank you!"
    assert confidence == MAGI_OCR_CONFIDENCE


def test_ties_go_to_magi():
    text, confidence = pick_better_ocr_result("abc", ("xyz", 0.9))
    assert text == "abc"
    assert confidence == MAGI_OCR_CONFIDENCE


def test_build_bubble_extractions_matches_bubble_detections_id_scheme():
    ocr_results = [("Thank you!", 0.85), ("Hello there.", 0.77)]
    extractions = build_bubble_extractions("page_04", ocr_results)

    assert extractions == [
        {"bubble_id": "page_04_b000", "dialogue_text": "Thank you!", "ocr_confidence": 0.85},
        {"bubble_id": "page_04_b001", "dialogue_text": "Hello there.", "ocr_confidence": 0.77},
    ]
    for extraction in extractions:
        BubbleExtraction(**extraction)  # validates against the Phase 3 schema


def test_build_bubble_extractions_with_no_bubbles():
    assert build_bubble_extractions("page_04", []) == []
