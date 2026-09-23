"""Runs the CV heuristic panel detector on a real sample page and checks the
result validates against PanelDetectionResult. The Magi backend is exercised
manually (it needs the ~2GB model download) rather than in this suite -- see
scripts/preview_panel_detection.py.
"""

from pathlib import Path

from comicreel.artifacts.schemas import PanelDetectionResult
from comicreel.stages.panel_detection import CVHeuristicPanelDetector

SAMPLE_PAGE = Path(__file__).resolve().parent.parent / "fixtures/sample_pages/page_04.jpg"


def test_cv_heuristic_output_matches_schema():
    stage = CVHeuristicPanelDetector()
    result = stage.run(
        "integration_test", "page_04", {"_input": {"image_path": str(SAMPLE_PAGE)}}
    )

    validated = PanelDetectionResult(**result)
    assert validated.page_id == "page_04"
    assert len(validated.panels) == 7  # hand-counted panels on this fixture page

    reading_orders = [p.reading_order_index for p in validated.panels]
    assert sorted(reading_orders) == list(range(len(validated.panels)))
