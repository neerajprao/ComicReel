"""One-off: run the Magi bubble detector on all 3 sample pages and save both
the raw JSON artifact and an overlay PNG (bubble box, tail, attribution line)
for each into `phase 2 output/`. Loads the model once and reuses it for all
3 pages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.config import get_backend_class
from comicreel.stages import bubble_detection  # noqa: F401  (registers backends)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PAGES = ["page_04", "page_05", "page_06"]
OUT_DIR = ROOT / "phase 2 output"


def _center(bbox):
    x, y, w, h = bbox
    return (int(x + w / 2), int(y + h / 2))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_cls = get_backend_class("bubble_detection", "magi")
    stage = stage_cls()

    for page_id in SAMPLE_PAGES:
        image_path = ROOT / "tests/fixtures/sample_pages" / f"{page_id}.jpg"
        result = stage.run("all_top_comics_6", page_id, {"_input": {"image_path": str(image_path)}})

        json_path = OUT_DIR / f"{page_id}_magi_bubbles.json"
        json_path.write_text(json.dumps(result, indent=2))

        image = cv2.imread(str(image_path))
        bubble_count = 0
        for panel in result["panels"]:
            for bubble in panel["bubbles"]:
                bubble_count += 1
                x, y, w, h = (int(v) for v in bubble["bbox"])
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 120, 0), 3)

                bubble_center = _center(bubble["bbox"])
                if bubble["tail_point"] is not None:
                    tail_point = (int(bubble["tail_point"][0]), int(bubble["tail_point"][1]))
                    cv2.circle(image, tail_point, 8, (0, 200, 0), -1)
                    cv2.line(image, bubble_center, tail_point, (0, 200, 0), 2)

                if bubble["attributed_character_bbox"] is not None:
                    char_center = _center(bubble["attributed_character_bbox"])
                    cv2.line(image, bubble_center, char_center, (0, 220, 255), 2)
                    cv2.circle(image, char_center, 6, (0, 220, 255), -1)

                label = f"{bubble['type'][0].upper()} {bubble['attribution_confidence']:.2f}"
                cv2.putText(image, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

        overlay_path = OUT_DIR / f"{page_id}_magi_bubbles_overlay.png"
        cv2.imwrite(str(overlay_path), image)

        print(f"{page_id}: {bubble_count} bubbles -> {json_path.name}, {overlay_path.name}")


if __name__ == "__main__":
    main()
