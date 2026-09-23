"""Print a page's VLM extraction result (dialogue + per-panel action/setting/
mood/onomatopoeia) in a readable form, for manual verification (see
progress.md Phase 3 verification criteria: dialogue vs. manual transcription
comparison, action-description plausibility check).

Usage:
    python scripts/preview_vlm_extraction.py <image_path> [backend]

backend defaults to qwen_vl_ollama (fast, needs `ollama serve` running with
qwen2.5vl:7b pulled); pass qwen_vl_hf to use the transformers-only fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.config import get_backend_class
from comicreel.stages import vlm_extraction  # noqa: F401  (registers backends)


def main(image_path: Path, backend: str = "qwen_vl_ollama") -> None:
    stage_cls = get_backend_class("vlm_extraction", backend)
    stage = stage_cls()
    result = stage.run("preview", image_path.stem, {"_input": {"image_path": str(image_path)}})

    for panel in result["panels"]:
        print(f"=== {panel['panel_id']} ===")
        print(f"  action:   {panel['action_description']}")
        print(f"  setting:  {panel['setting']}")
        print(f"  mood:     {panel['mood']}")
        print(f"  sfx:      {panel['onomatopoeia']}")
        for bubble in panel["bubbles"]:
            print(f"    [{bubble['ocr_confidence']:.2f}] {bubble['dialogue_text']!r}")
        print()


if __name__ == "__main__":
    backend_arg = sys.argv[2] if len(sys.argv) > 2 else "qwen_vl_ollama"
    main(Path(sys.argv[1]), backend_arg)
