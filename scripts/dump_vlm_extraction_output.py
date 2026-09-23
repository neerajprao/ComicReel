"""One-off: run the Qwen2.5-VL (Ollama) extractor on all 3 sample pages and
save the raw JSON artifact for each into `phase 3 output/`, plus a plain-text
summary readable without parsing JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.config import get_backend_class
from comicreel.stages import vlm_extraction  # noqa: F401  (registers backends)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PAGES = ["page_04", "page_05", "page_06"]
OUT_DIR = ROOT / "phase 3 output"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_cls = get_backend_class("vlm_extraction", "qwen_vl_ollama")
    stage = stage_cls()

    for page_id in SAMPLE_PAGES:
        image_path = ROOT / "tests/fixtures/sample_pages" / f"{page_id}.jpg"
        result = stage.run("all_top_comics_6", page_id, {"_input": {"image_path": str(image_path)}})

        json_path = OUT_DIR / f"{page_id}_vlm_extraction.json"
        json_path.write_text(json.dumps(result, indent=2))

        lines = []
        for panel in result["panels"]:
            lines.append(f"=== {panel['panel_id']} ===")
            lines.append(f"  action:   {panel['action_description']}")
            lines.append(f"  setting:  {panel['setting']}")
            lines.append(f"  mood:     {panel['mood']}")
            lines.append(f"  sfx:      {panel['onomatopoeia']}")
            for bubble in panel["bubbles"]:
                lines.append(f"    [{bubble['ocr_confidence']:.2f}] {bubble['dialogue_text']!r}")
            lines.append("")
        summary_path = OUT_DIR / f"{page_id}_vlm_extraction.txt"
        summary_path.write_text("\n".join(lines))

        print(f"{page_id}: {len(result['panels'])} panels -> {json_path.name}, {summary_path.name}")


if __name__ == "__main__":
    main()
