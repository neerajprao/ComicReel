"""One-off: run a character_embedding backend across all 3 sample pages (same
comic_id, so clustering accumulates across pages like a real multi-page run)
and save the raw JSON artifact for each into `phase 4 output/`, plus a plain-
text summary readable without parsing JSON.

Usage:
    python scripts/dump_character_embedding_output.py [clip|dinov2]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.config import get_backend_class
from comicreel.stages import character_embedding  # noqa: F401  (registers backends)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PAGES = ["page_04", "page_05", "page_06"]
COMIC_ID = "all_top_comics_6"
OUT_DIR = ROOT / "phase 4 output"


def main(backend_name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_cls = get_backend_class("character_embedding", backend_name)
    stage = stage_cls(options={"cache_root": str(ROOT / "data" / "cache")})

    for page_id in SAMPLE_PAGES:
        image_path = ROOT / "tests/fixtures/sample_pages" / f"{page_id}.jpg"
        result = stage.run(COMIC_ID, page_id, {"_input": {"image_path": str(image_path)}})

        json_path = OUT_DIR / f"{page_id}_{backend_name}_characters.json"
        json_path.write_text(json.dumps(result, indent=2))

        lines = [f"=== known characters so far: {len(result['characters'])} ==="]
        for character in result["characters"]:
            lines.append(f"  {character['character_id']}  name={character['name']!r}")
        lines.append("")
        lines.append(f"=== {page_id} assignments ===")
        for assignment in result["panel_assignments"]:
            lines.append(
                f"  {assignment['panel_id']}: {assignment['character_id']} "
                f"(similarity={assignment['similarity']:.2f})"
            )
        summary_path = OUT_DIR / f"{page_id}_{backend_name}_characters.txt"
        summary_path.write_text("\n".join(lines))

        print(
            f"{page_id}: {len(result['characters'])} known characters, "
            f"{len(result['panel_assignments'])} assignments -> {json_path.name}"
        )


if __name__ == "__main__":
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "clip"
    main(backend_name)
