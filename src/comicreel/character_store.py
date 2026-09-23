"""Per-comic, cross-page persistent store of character clusters.

Character identity tracking is the pipeline's one inherently cross-page
stage: a character's identity spans the whole comic, not a single page, but
the orchestrator (`pipeline/orchestrator.py`) runs one page at a time with no
cross-page state of its own. This module is that missing state: a small JSON
file per comic, keyed by embedder backend (CLIP and DINOv2 embeddings aren't
comparable, so each backend gets its own independent set of clusters),
updated incrementally as each page's character_embedding stage runs.

Kept deliberately separate from `artifacts/cache.py`'s `StageCache`, which is
keyed by (comic_id, page_id, stage, input_hash) and always overwritten
wholesale -- this store is instead read-modify-written incrementally across
many pages, which is a different access pattern.

`characters.yaml` (optional, user-editable, living alongside a comic's raw
scans) lets a user override a cluster's auto-generated `character_id` with a
real name once they've reviewed the clusters (see
`scripts/preview_character_clusters.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from comicreel.character_clustering import assign_embedding, assign_embedding_complete_linkage

DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_RAW_SCANS_ROOT = Path("data/raw_scans")


def store_path(comic_id: str, cache_root: Path | str, backend_name: str) -> Path:
    return Path(cache_root) / comic_id / "characters" / backend_name / "index.json"


def load_store(comic_id: str, cache_root: Path | str, backend_name: str) -> dict[str, Any]:
    path = store_path(comic_id, cache_root, backend_name)
    if path.exists():
        return json.loads(path.read_text())
    return {"clusters": []}


def save_store(
    comic_id: str, cache_root: Path | str, backend_name: str, store: dict[str, Any]
) -> Path:
    path = store_path(comic_id, cache_root, backend_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))
    return path


def _centroid(members: list[list[float]]) -> list[float]:
    return np.mean(np.asarray(members, dtype=float), axis=0).tolist()


def assign_character(
    store: dict[str, Any],
    embedding: list[float],
    threshold: float,
    mode: str = "centroid",
) -> tuple[str, float, bool]:
    """Match `embedding` against `store`'s existing clusters, mutating
    `store` in place (adds `embedding` to the matched cluster's member list,
    or appends a brand-new cluster). Returns (character_id, similarity, is_new).

    Every cluster's full embedding history is kept in `members` (not just a
    running centroid), so either matching rule can be used without changing
    the store's shape:
    - `mode="centroid"` (default): `embedding` just has to clear `threshold`
      against the cluster's mean -- looser, prone to under-splitting when a
      cluster's average drifts toward a "generic" point that several
      different characters happen to resemble (see the Phase 4 decisions
      log for a measured example on this project's own sample pages).
    - `mode="complete_linkage"`: `embedding` must clear `threshold` against
      *every* existing member -- stricter, trades under-splitting for
      over-splitting (more, smaller, purer clusters that a human then merges
      via `characters.yaml`).

    A new cluster's `representative_crop` is left `None` -- the caller sets
    it once it knows where the crop image was saved (see
    `stages/character_embedding.py`), since this function has no notion of
    crop files at all.
    """
    clusters = store["clusters"]
    if mode == "complete_linkage":
        member_lists = [c["members"] for c in clusters]
        idx, similarity = assign_embedding_complete_linkage(embedding, member_lists, threshold)
    else:
        centroids = [_centroid(c["members"]) for c in clusters]
        idx, similarity = assign_embedding(embedding, centroids, threshold)

    if idx is not None:
        cluster = clusters[idx]
        cluster["members"].append(list(embedding))
        cluster["count"] += 1
        return cluster["character_id"], similarity, False

    character_id = f"char_{len(clusters):03d}"
    clusters.append(
        {
            "character_id": character_id,
            "members": [list(embedding)],
            "count": 1,
            "representative_crop": None,
            # Per-member crop file paths, parallel to `members` -- the
            # caller (stages/character_embedding.py) appends to this after
            # every assign_character call, matched or new. Lets a later
            # verification pass (character_verification.py) locate the exact
            # crop image behind any member index it wants to show a VLM.
            "crop_paths": [],
        }
    )
    return character_id, similarity, True


def load_character_names(
    comic_id: str, raw_scans_root: Path | str = DEFAULT_RAW_SCANS_ROOT
) -> dict[str, str]:
    """Read `<raw_scans_root>/<comic_id>/characters.yaml` (character_id ->
    name), or an empty mapping if the user hasn't created one yet."""
    path = Path(raw_scans_root) / comic_id / "characters.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}
