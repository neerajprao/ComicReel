"""Runs the Magi bubble detector on a real sample page and checks the result
validates against BubbleDetectionResult. Needs the ~2GB Magi model, so this
is slower than the rest of the suite -- run explicitly when touching
bubble_detection, not expected to run in a tight edit loop.
"""

from pathlib import Path

from comicreel.artifacts.schemas import BubbleDetectionResult
from comicreel.stages.bubble_detection import MagiBubbleDetector

SAMPLE_PAGE = Path(__file__).resolve().parent.parent / "fixtures/sample_pages/page_04.jpg"


def test_magi_bubble_output_matches_schema():
    stage = MagiBubbleDetector()
    result = stage.run(
        "integration_test", "page_04", {"_input": {"image_path": str(SAMPLE_PAGE)}}
    )

    validated = BubbleDetectionResult(**result)
    assert validated.page_id == "page_04"
    assert len(validated.panels) == 7  # matches panel_detection's panel count on this fixture

    total_bubbles = sum(len(p.bubbles) for p in validated.panels)
    assert total_bubbles > 0  # this page has visible dialogue in every panel

    for panel in validated.panels:
        for bubble in panel.bubbles:
            assert bubble.type in ("speech", "thought", "narration", "sfx")
            assert 0.0 <= bubble.attribution_confidence <= 1.0
