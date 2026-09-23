"""Runs the Qwen2.5-VL extractor on a real sample page and checks the result
validates against VLMExtractionResult. Needs a running Ollama server with
qwen2.5vl:7b pulled, so this is much slower than the rest of the suite --
run explicitly when touching vlm_extraction, not expected in a tight edit
loop.
"""

from pathlib import Path

from comicreel.artifacts.schemas import VLMExtractionResult
from comicreel.stages.vlm_extraction import QwenVLOllamaExtractor

SAMPLE_PAGE = Path(__file__).resolve().parent.parent / "fixtures/sample_pages/page_04.jpg"


def test_qwen_vl_ollama_output_matches_schema():
    stage = QwenVLOllamaExtractor()
    result = stage.run(
        "integration_test", "page_04", {"_input": {"image_path": str(SAMPLE_PAGE)}}
    )

    validated = VLMExtractionResult(**result)
    assert validated.page_id == "page_04"
    assert len(validated.panels) == 7  # matches panel_detection's panel count on this fixture

    for panel in validated.panels:
        assert panel.action_description.strip() != ""
        assert isinstance(panel.onomatopoeia, list)

    total_bubbles = sum(len(p.bubbles) for p in validated.panels)
    assert total_bubbles > 0
