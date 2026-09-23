"""Draw detected panel boxes + reading-order numbers on a page image, for
manual visual verification (see progress.md Phase 1 verification criteria).

Usage:
    python scripts/preview_panel_detection.py <backend> <image_path> [output_path]

backend is a registered panel_detection backend name, e.g. cv_heuristic or magi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.config import get_backend_class
from comicreel.stages import panel_detection  # noqa: F401  (registers backends)


def draw_overlay(image_path: Path, backend_name: str, output_path: Path) -> None:
    stage_cls = get_backend_class("panel_detection", backend_name)
    stage = stage_cls()
    result = stage.run("preview", image_path.stem, {"_input": {"image_path": str(image_path)}})

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"{len(result['panels'])} panels -> {output_path}")


if __name__ == "__main__":
    backend_name = sys.argv[1]
    image_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else image_path.with_name(
        f"{image_path.stem}_{backend_name}_overlay.png"
    )
    draw_overlay(image_path, backend_name, output_path)
