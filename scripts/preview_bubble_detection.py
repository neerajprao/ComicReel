"""Draw detected bubbles (blue), tail points (green dot), and speaker
attribution lines (yellow, bubble center -> attributed character center) on
a page image, for manual visual verification (see progress.md Phase 2
verification criteria).

Usage:
    python scripts/preview_bubble_detection.py <backend> <image_path> [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.config import get_backend_class
from comicreel.stages import bubble_detection  # noqa: F401  (registers backends)


def _center(bbox):
    x, y, w, h = bbox
    return (int(x + w / 2), int(y + h / 2))


def draw_overlay(image_path: Path, backend_name: str, output_path: Path) -> None:
    stage_cls = get_backend_class("bubble_detection", backend_name)
    stage = stage_cls()
    result = stage.run("preview", image_path.stem, {"_input": {"image_path": str(image_path)}})

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"{bubble_count} bubbles -> {output_path}")


if __name__ == "__main__":
    backend_name = sys.argv[1]
    image_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else image_path.with_name(
        f"{image_path.stem}_{backend_name}_bubbles_overlay.png"
    )
    draw_overlay(image_path, backend_name, output_path)
