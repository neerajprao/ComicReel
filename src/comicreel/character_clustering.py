"""Incremental cosine-similarity clustering of character embeddings across a comic.

Pure, backend-agnostic functions (no model/backend-specific code) so any
embedder (CLIP, DINOv2, ...) can reuse the same clustering logic -- same
"detections from the backend, meaning from our own code" pattern as
reading_order.py and speaker_attribution.py.

Clustering itself is greedy and incremental, not a batch algorithm: each new
embedding is compared against the existing cluster centroids so far and
either joins the best-matching one (if similarity clears `threshold`) or
starts a new cluster. This matches how pages are processed one at a time by
the orchestrator, with no full-comic embedding set available up front.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def best_match(
    embedding: Sequence[float], centroids: Sequence[Sequence[float]]
) -> tuple[int | None, float]:
    """Index of the centroid most similar to `embedding`, and that
    similarity. (None, 0.0) if `centroids` is empty."""
    if not centroids:
        return None, 0.0
    similarities = [cosine_similarity(embedding, c) for c in centroids]
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    return best_idx, similarities[best_idx]


def update_centroid(
    centroid: Sequence[float], member_count: int, new_embedding: Sequence[float]
) -> list[float]:
    """Running-mean update of a cluster centroid given one more member."""
    centroid_arr = np.asarray(centroid, dtype=float)
    new_arr = np.asarray(new_embedding, dtype=float)
    updated = (centroid_arr * member_count + new_arr) / (member_count + 1)
    return updated.tolist()


def assign_embedding(
    embedding: Sequence[float], centroids: Sequence[Sequence[float]], threshold: float
) -> tuple[int | None, float]:
    """Decide which existing cluster (by index into `centroids`) `embedding`
    belongs to. Returns (None, best_similarity) if it should start a new
    cluster instead -- either no clusters exist yet, or the best match is
    below `threshold`."""
    idx, similarity = best_match(embedding, centroids)
    if idx is not None and similarity >= threshold:
        return idx, similarity
    return None, similarity


def complete_linkage_similarity(
    embedding: Sequence[float], members: Sequence[Sequence[float]]
) -> float:
    """The *minimum* similarity between `embedding` and every existing
    member of a cluster -- a strict "must resemble everyone already in the
    group" rule, unlike `best_match`'s single running centroid (which only
    has to resemble the group's average). Trades under-splitting (centroid
    mode's failure mode: dissimilar items sharing a cluster because each was
    close enough to a drifting average) for over-splitting (more, smaller,
    purer clusters). 0.0 for an empty cluster."""
    if not members:
        return 0.0
    return min(cosine_similarity(embedding, member) for member in members)


def best_complete_linkage_match(
    embedding: Sequence[float], clusters: Sequence[Sequence[Sequence[float]]]
) -> tuple[int | None, float]:
    """Index of the cluster (given as a list of member-lists) `embedding`
    fits best under complete-linkage, and that score. (None, 0.0) if
    `clusters` is empty."""
    if not clusters:
        return None, 0.0
    scores = [complete_linkage_similarity(embedding, members) for members in clusters]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, scores[best_idx]


def assign_embedding_complete_linkage(
    embedding: Sequence[float], clusters: Sequence[Sequence[Sequence[float]]], threshold: float
) -> tuple[int | None, float]:
    """Complete-linkage counterpart to `assign_embedding`: decide which
    existing cluster `embedding` belongs to by requiring it clear
    `threshold` against every one of that cluster's members, not just its
    centroid."""
    idx, score = best_complete_linkage_match(embedding, clusters)
    if idx is not None and score >= threshold:
        return idx, score
    return None, score
