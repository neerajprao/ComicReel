"""One-off: run the Magi panel detector on all 3 sample pages and save both
the raw JSON artifact and an overlay PNG (bbox + reading-order number) for
each into `phase 1 output/`. Loads the model once and reuses it for all 3
pages (loading is the slow part).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.config import get_backend_class
from comicreel.stages import panel_detection  # noqa: F401  (registers backends)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PAGES = ["page_04", "page_05", "page_06"]
OUT_DIR = ROOT / "phase 1 output"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_cls = get_backend_class("panel_detection", "magi")
    stage = stage_cls()

    for page_id in SAMPLE_PAGES:
        image_path = ROOT / "tests/fixtures/sample_pages" / f"{page_id}.jpg"
        result = stage.run("all_top_comics_6", page_id, {"_input": {"image_path": str(image_path)}})

        json_path = OUT_DIR / f"{page_id}_magi.json"
        json_path.write_text(json.dumps(result, indent=2))

        image = cv2.imread(str(image_path))
        for panel in result["panels"]:
            x, y, w, h = (int(v) for v in panel["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(
                image,
                str(panel["reading_order_index"]),
                (x + 8, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255),
                3,
            )
        overlay_path = OUT_DIR / f"{page_id}_magi_overlay.png"
        cv2.imwrite(str(overlay_path), image)

        print(f"{page_id}: {len(result['panels'])} panels -> {json_path.name}, {overlay_path.name}")


if __name__ == "__main__":
    main()
