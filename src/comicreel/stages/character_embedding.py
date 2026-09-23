"""Character-embedding stage backends: CLIP and DINOv2.

Returns a dict matching `CharacterIndex` (see `comicreel.artifacts.schemas`):
the full character registry clustered so far for this comic, plus this
page's panel_assignments (which detected character crop maps to which
character_id, and how confidently).

Both backends make their own Magi call for per-page character bboxes (same
"own detections, independent of sibling stages" pattern as panel_detection /
bubble_detection / vlm_extraction -- see the Phase 4 decisions log in
progress.md: which stage should own character-box detection long-term is
still an open question, deliberately left as the safe, reversible default for
now). Each detected crop is embedded, then incrementally clustered against a
persistent per-comic, per-backend store (`character_store.py`) using cosine
similarity (`character_clustering.py`) -- CLIP and DINOv2 embeddings aren't
comparable, so switching backends starts a fresh set of clusters rather than
silently mixing embedding spaces.

Every crop (not just the first one per cluster) is saved to disk, named
`<character_id>__<page_id>_c<i>.png`, so a human reviewer can later dump a
crop-grid per character_id for visual verification (see
`scripts/preview_character_clusters.py`) -- not just the one representative
crop kept in the schema's `Character.representative_crop`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from comicreel.character_color import color_histogram
from comicreel.character_crop import head_crop_bbox
from comicreel.character_store import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_RAW_SCANS_ROOT,
    assign_character,
    load_character_names,
    load_store,
    save_store,
)
from comicreel.config import register_backend
from comicreel.geometry import assign_to_panel, to_bbox
from comicreel.pipeline.stage_base import Stage
from comicreel.stages.character_embedders import (
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_DINOV2_MODEL_ID,
    embed_image_clip,
    embed_image_dinov2,
)
from comicreel.stages.magi_model import run_magi_detection


def _fuse_embedding_and_color(
    embedding: np.ndarray, color_hist: list[float], color_weight: float
) -> np.ndarray:
    """Combine an appearance embedding with a color histogram into one
    vector, so the existing cosine-similarity clustering code
    (`character_clustering.py`) sees a single fused signal without any
    changes to it: L2-normalize each part, scale so their squared norms sum
    to 1, and concatenate -- then cosine similarity of two fused vectors is
    approximately `(1 - color_weight)` times the embeddings' cosine
    similarity plus `color_weight` times the histograms' cosine similarity.

    A Phase 4 measurement found DINOv2 alone confuses characters that a
    human tells apart mostly by costume/hair color (see character_color.py's
    module docstring) -- this is that fix. `color_weight=0` recovers plain
    embedding similarity exactly (used by tests that don't care about color).
    """
    if color_weight <= 0:
        return embedding

    embedding_arr = np.asarray(embedding, dtype=float)
    color_arr = np.asarray(color_hist, dtype=float)
    embedding_norm = embedding_arr / (np.linalg.norm(embedding_arr) or 1.0)
    color_norm = color_arr / (np.linalg.norm(color_arr) or 1.0)
    return np.concatenate(
        [embedding_norm * np.sqrt(1 - color_weight), color_norm * np.sqrt(color_weight)]
    )


def _embed_and_cluster(
    *,
    comic_id: str,
    page_id: str,
    image_path: str,
    device: str,
    options: dict[str, Any],
    embed_fn: Any,
    model_id: str,
    backend_name: str,
) -> dict[str, Any]:
    cache_root = Path(options.get("cache_root", DEFAULT_CACHE_ROOT))
    raw_scans_root = Path(options.get("raw_scans_root", DEFAULT_RAW_SCANS_ROOT))
    # Defaults mirror configs/pipeline.yaml's measured recommendation (see the
    # Phase 4 decisions log): plain centroid clustering on full character
    # boxes under-split badly on real stylized-comic crops, so
    # complete_linkage + head crops is the sane out-of-the-box behavior even
    # if a caller passes no options at all.
    similarity_threshold = options.get("similarity_threshold", 0.55)
    clustering_mode = options.get("clustering_mode", "complete_linkage")

    magi_result = run_magi_detection(
        image_path,
        device,
        panel_detection_threshold=options.get("panel_detection_threshold", 0.2),
        character_detection_threshold=options.get("character_detection_threshold", 0.3),
    )
    panel_boxes = [to_bbox(*p) for p in magi_result["panels"]]
    character_boxes = [to_bbox(*c) for c in magi_result["characters"]]

    page_image = Image.open(image_path).convert("RGB")
    store = load_store(comic_id, cache_root, backend_name)
    crop_dir = cache_root / comic_id / "characters" / backend_name / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    use_head_crop = options.get("use_head_crop", True)
    head_crop_top_fraction = options.get("head_crop_top_fraction", 0.45)
    head_crop_width_fraction = options.get("head_crop_width_fraction", 0.7)
    # 0.25 chosen from measured comparisons on limited data (see the Phase 4
    # decisions log) -- no single weight cleanly dominated in that test, so
    # this isn't a validated optimum, just a reasonable middle ground before
    # the VLM verification pass (character_verification.py) catches whatever
    # this and the embedding still get wrong.
    color_weight = options.get("color_weight", 0.25)

    assignments: list[dict[str, Any]] = []
    for i, bbox in enumerate(character_boxes):
        if not panel_boxes:
            continue  # CharacterAssignment.panel_id is required; nothing to assign to

        # Embed just the head/face region, not the whole character box -- a
        # Phase 4 measurement found the full box's clothing/pose/background
        # swamps the identity signal CLIP/DINOv2 need to tell characters
        # apart (see character_crop.py and the decisions log).
        crop_box = (
            head_crop_bbox(bbox, top_fraction=head_crop_top_fraction, width_fraction=head_crop_width_fraction)
            if use_head_crop
            else bbox
        )
        x, y, w, h = (int(v) for v in crop_box)
        crop = page_image.crop((x, y, x + w, y + h))
        if crop.width == 0 or crop.height == 0:
            continue

        # Color comes from the *whole* character box (costume + hair), not
        # the head-only crop used for the embedding -- clothing color is
        # often the strongest identity signal in flat-colored comic art, and
        # the head crop alone would miss most of it.
        full_x, full_y, full_w, full_h = (int(v) for v in bbox)
        full_crop = page_image.crop((full_x, full_y, full_x + full_w, full_y + full_h))
        color_hist = color_histogram(full_crop) if full_crop.width and full_crop.height else None

        embedding = embed_fn(crop, device=device, model_id=model_id)
        fused = (
            _fuse_embedding_and_color(embedding, color_hist, color_weight)
            if color_hist is not None
            else embedding
        )
        character_id, similarity, is_new = assign_character(
            store, fused.tolist(), similarity_threshold, mode=clustering_mode
        )

        crop_path = crop_dir / f"{character_id}__{page_id}_c{i:03d}.png"
        crop.save(crop_path)
        cluster = next(c for c in store["clusters"] if c["character_id"] == character_id)
        cluster.setdefault("crop_paths", []).append(str(crop_path))
        if is_new:
            cluster["representative_crop"] = str(crop_path)

        panel_idx = assign_to_panel(bbox, panel_boxes)
        assignments.append(
            {
                "panel_id": f"{page_id}_p{panel_idx:03d}",
                "bubble_id": None,
                "character_id": character_id,
                "similarity": similarity,
            }
        )

    save_store(comic_id, cache_root, backend_name, store)
    names = load_character_names(comic_id, raw_scans_root)

    characters = [
        {
            "character_id": c["character_id"],
            "representative_crop": c["representative_crop"],
            "name": names.get(c["character_id"]),
        }
        for c in store["clusters"]
    ]
    return {"characters": characters, "panel_assignments": assignments}


@register_backend("character_embedding", "clip")
class ClipCharacterEmbedder(Stage):
    name = "clip"

    def __init__(self, options: dict[str, Any] | None = None, device: str = "cpu"):
        self.options = options or {}
        self.device = device

    def run(self, comic_id: str, page_id: str, prior_artifacts: dict[str, Any]) -> dict[str, Any]:
        image_path = prior_artifacts["_input"]["image_path"]
        return _embed_and_cluster(
            comic_id=comic_id,
            page_id=page_id,
            image_path=image_path,
            device=self.device,
            options=self.options,
            embed_fn=embed_image_clip,
            model_id=self.options.get("clip_model_id", DEFAULT_CLIP_MODEL_ID),
            backend_name="clip",
        )


@register_backend("character_embedding", "dinov2")
class DinoV2CharacterEmbedder(Stage):
    name = "dinov2"

    def __init__(self, options: dict[str, Any] | None = None, device: str = "cpu"):
        self.options = options or {}
        self.device = device

    def run(self, comic_id: str, page_id: str, prior_artifacts: dict[str, Any]) -> dict[str, Any]:
        image_path = prior_artifacts["_input"]["image_path"]
        return _embed_and_cluster(
            comic_id=comic_id,
            page_id=page_id,
            image_path=image_path,
            device=self.device,
            options=self.options,
            embed_fn=embed_image_dinov2,
            model_id=self.options.get("dinov2_model_id", DEFAULT_DINOV2_MODEL_ID),
            backend_name="dinov2",
        )
