"""Build one crop-grid image per character cluster, for manual visual
verification that the clustering hasn't merged two different characters or
split one character into two clusters (see progress.md Phase 4 verification
criteria: "crop-grid dump per cluster for human visual review").

Usage:
    python scripts/preview_character_clusters.py <comic_id> <backend_name> [output_dir]

Run `scripts/dump_character_embedding_output.py` first to populate the
per-comic character store this script reads from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.character_store import load_store

ROOT = Path(__file__).resolve().parent.parent
TILE_SIZE = 128
COLS = 6


def _load_tile(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        return np.full((TILE_SIZE, TILE_SIZE, 3), 64, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(TILE_SIZE / w, TILE_SIZE / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    tile = np.full((TILE_SIZE, TILE_SIZE, 3), 32, dtype=np.uint8)
    y0 = (TILE_SIZE - resized.shape[0]) // 2
    x0 = (TILE_SIZE - resized.shape[1]) // 2
    tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return tile


def build_grid(crop_paths: list[Path]) -> np.ndarray:
    tiles = [_load_tile(p) for p in crop_paths]
    rows = [tiles[i : i + COLS] for i in range(0, len(tiles), COLS)]
    row_images = []
    for row in rows:
        row = row + [np.full((TILE_SIZE, TILE_SIZE, 3), 0, dtype=np.uint8)] * (COLS - len(row))
        row_images.append(np.hstack(row))
    return np.vstack(row_images)


def main(comic_id: str, backend_name: str, out_dir: Path) -> None:
    cache_root = ROOT / "data" / "cache"
    store = load_store(comic_id, cache_root, backend_name)
    crop_dir = cache_root / comic_id / "characters" / backend_name / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not store["clusters"]:
        print(f"No character clusters found for {comic_id}/{backend_name} in {cache_root}")
        return

    for cluster in store["clusters"]:
        character_id = cluster["character_id"]
        crop_paths = sorted(crop_dir.glob(f"{character_id}__*.png"))
        if not crop_paths:
            continue
        grid = build_grid(crop_paths)
        out_path = out_dir / f"{comic_id}_{backend_name}_{character_id}_grid.png"
        cv2.imwrite(str(out_path), grid)
        print(f"{character_id}: {len(crop_paths)} crops -> {out_path.name}")


if __name__ == "__main__":
    comic_id = sys.argv[1]
    backend_name = sys.argv[2]
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "phase 4 output"
    main(comic_id, backend_name, out_dir)
