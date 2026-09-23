"""VLM-based verification pass on top of embedding clustering.

A Phase 4 measurement found DINOv2+color-fusion clustering alone still
confuses some characters (see progress.md's decisions log). A second
measurement found a local VLM (Qwen2.5-VL) is much better at this specific
judgment when asked directly, pairwise, with a feature-decomposition prompt
("list hair/clothing/accessories, then decide") -- but *only* pairwise; asking
it to self-organize many crops at once in one shot failed outright. This
module is the targeted middle ground: instead of checking every possible
pair (O(n^2), too slow), it only asks the VLM about the pairs embedding
clustering was *least* sure about:

- Within a cluster: its weakest-linked pair (the two members least similar
  to each other) -- if the VLM says these are different characters, the
  cluster was a false merge and should split.
- Between two clusters: their strongest-linked pair (the two members most
  similar to each other) -- if the VLM says these are the same character,
  the clusters were a missed merge and should combine.

This keeps the number of VLM calls proportional to the number of clusters,
not the number of crops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from comicreel.character_clustering import cosine_similarity

# Only bother asking the VLM about a cluster's weakest-linked pair if that
# link already looks shaky -- a confidently coherent cluster (high internal
# similarity) doesn't need checking. Symmetric gate for merges: only check a
# cross-cluster pair if it's already plausible, not any random pair.
DEFAULT_SPLIT_CANDIDATE_MAX_SIM = 0.6
# Kept close to (just below) the clustering stage's own similarity_threshold
# (0.55, see stages/character_embedding.py) -- a pair below this wasn't
# close to being merged by the embedding in the first place, so treating it
# as a "plausible missed merge" candidate is asking the VLM to adjudicate
# pairs with almost no real signal behind them. A benchmark run with this
# set too low (0.35) fed the VLM many weak, near-arbitrary pairs and got a
# high false "same" rate back -- see the Phase 4 decisions log.
DEFAULT_MERGE_CANDIDATE_MIN_SIM = 0.5


def weakest_intra_cluster_pair(
    members: list[list[float]],
) -> tuple[int, int, float] | None:
    """Indices (into `members`) of the least-similar pair within one
    cluster, and their similarity -- the pair most likely to reveal a false
    merge. None if the cluster has fewer than 2 members."""
    if len(members) < 2:
        return None
    worst_pair, worst_sim = None, 2.0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            sim = cosine_similarity(members[i], members[j])
            if sim < worst_sim:
                worst_pair, worst_sim = (i, j), sim
    return (*worst_pair, worst_sim)


def strongest_inter_cluster_pair(
    members_a: list[list[float]], members_b: list[list[float]]
) -> tuple[int, int, float] | None:
    """Indices (i into `members_a`, j into `members_b`) of the most-similar
    pair across two different clusters, and their similarity -- the pair
    most likely to reveal a missed merge. None if either cluster is empty."""
    if not members_a or not members_b:
        return None
    best_pair, best_sim = None, -2.0
    for i, a in enumerate(members_a):
        for j, b in enumerate(members_b):
            sim = cosine_similarity(a, b)
            if sim > best_sim:
                best_pair, best_sim = (i, j), sim
    return (*best_pair, best_sim)


VERIFICATION_PROMPT = """These two images are crops from a comic book, each showing one character.

First, briefly describe each image's hair color/style, clothing color/style, skin tone, and any accessories (hats, crowns, glasses) -- one short line per image.

Then, based ONLY on that comparison, decide: do these two crops show the SAME individual character (even if drawn in a different pose or panel), or two DIFFERENT characters?

Respond with ONLY a JSON object in exactly this format, no other text:
{"image_1": "short feature description", "image_2": "short feature description", "same_character": true or false}"""


# Ollama's Qwen2.5-VL image processor hard-crashes the whole server
# (`height/width must be larger than factor:28`) on any image with a
# dimension under 28px -- confirmed from its own panic log after it took
# down a benchmark run on a comic sourced from a low-resolution scan, where
# some head-crops came out smaller than this. Upscaling first avoids ever
# sending it a dimension it can't handle.
_MIN_VLM_IMAGE_DIM = 28


def _ensure_min_size(image_path: str) -> bytes:
    import io

    from PIL import Image

    image = Image.open(image_path)
    if min(image.size) >= _MIN_VLM_IMAGE_DIM:
        with open(image_path, "rb") as f:
            return f.read()

    scale = _MIN_VLM_IMAGE_DIM / min(image.size)
    new_size = (max(_MIN_VLM_IMAGE_DIM, round(image.width * scale)), max(_MIN_VLM_IMAGE_DIM, round(image.height * scale)))
    resized = image.convert("RGB").resize(new_size)
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def verify_same_character(image_a_path: str, image_b_path: str, model: str) -> tuple[bool, dict[str, Any]]:
    """Ask the VLM whether two character crops show the same individual.
    Returns (same_character, raw_parsed_response)."""
    import json
    import re

    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": VERIFICATION_PROMPT,
                "images": [_ensure_min_size(image_a_path), _ensure_min_size(image_b_path)],
            }
        ],
        options={"temperature": 0},
    )
    raw = response["message"]["content"]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in VLM verification response: {raw!r}")
    parsed = json.loads(match.group(0))
    return bool(parsed["same_character"]), parsed


VerifyFn = Callable[[str, str], bool]


def refine_clusters(
    clusters: list[dict[str, Any]],
    verify_fn: VerifyFn,
    *,
    split_candidate_max_sim: float = DEFAULT_SPLIT_CANDIDATE_MAX_SIM,
    merge_candidate_min_sim: float = DEFAULT_MERGE_CANDIDATE_MIN_SIM,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the split/merge verification pass to a character store's
    `clusters` list (each a dict with `character_id`, `members`,
    `crop_paths`, `count`, `representative_crop`; see character_store.py).

    `verify_fn(crop_path_a, crop_path_b) -> bool` is injected rather than
    calling Ollama directly, so this logic is unit-testable without a
    running VLM. Returns (refined_clusters, log) -- log is a list of dicts
    describing every check made and every split/merge actually applied, for
    reporting.

    Splitting a cluster reassigns each member to whichever of the two
    disagreeing "seed" members (the weakest-linked pair the VLM said were
    different) it's more similar to -- a simple 2-way split, not a general
    re-clustering, since the weakest link is exactly the boundary a false
    merge happened at.
    """
    clusters = [dict(c) for c in clusters]  # shallow-copy the list of dicts we'll mutate
    log: list[dict[str, Any]] = []

    to_split: list[tuple[int, int, int]] = []
    for idx, cluster in enumerate(clusters):
        result = weakest_intra_cluster_pair(cluster["members"])
        if result is None:
            continue
        i, j, sim = result
        if sim > split_candidate_max_sim:
            continue
        same = verify_fn(cluster["crop_paths"][i], cluster["crop_paths"][j])
        log.append(
            {"type": "split_check", "character_id": cluster["character_id"], "sim": sim, "same": same}
        )
        if not same:
            to_split.append((idx, i, j))

    next_id = len(clusters)
    for idx, seed_i, seed_j in to_split:
        cluster = clusters[idx]
        members, crop_paths = cluster["members"], cluster["crop_paths"]
        seed_a, seed_b = members[seed_i], members[seed_j]
        group_a, group_b = [], []
        for k in range(len(members)):
            target = group_a if cosine_similarity(members[k], seed_a) >= cosine_similarity(
                members[k], seed_b
            ) else group_b
            target.append(k)

        new_character_id = f"char_{next_id:03d}"
        next_id += 1
        clusters.append(
            {
                "character_id": new_character_id,
                "members": [members[k] for k in group_b],
                "crop_paths": [crop_paths[k] for k in group_b],
                "count": len(group_b),
                "representative_crop": crop_paths[group_b[0]],
            }
        )
        cluster["members"] = [members[k] for k in group_a]
        cluster["crop_paths"] = [crop_paths[k] for k in group_a]
        cluster["count"] = len(group_a)
        cluster["representative_crop"] = crop_paths[group_a[0]]
        log.append(
            {
                "type": "split_applied",
                "from": cluster["character_id"],
                "to": new_character_id,
                "sizes": [len(group_a), len(group_b)],
            }
        )

    # Every candidate pair is checked against the ORIGINAL (post-split, but
    # pre-merge) member lists, snapshotted here. A first fix attempt used
    # union-find to combine every cluster transitively connected by a chain
    # of verified "same" pairs (A-B same, B-C same => merge A+B+C) -- a
    # second Phase 4 benchmark run showed this is unsafe: with the VLM's
    # real per-check accuracy (~85-90%, measured earlier), transitive
    # closure means a SINGLE false "same" verdict anywhere in the candidate
    # graph can weld unrelated clusters together, and as more clusters get
    # checked the odds of hitting at least one such edge approach certainty.
    # Two different attempts each collapsed whole comics down to one or two
    # giant clusters. The fix is architectural, not a threshold tweak: each
    # cluster may take part in **at most one merge per pass** (a cluster,
    # once merged, is removed from further consideration this pass) --
    # bounding the worst-case damage of any single wrong verdict to
    # combining exactly two clusters, never a chain. A character split
    # across more than two clusters may need more than one refinement pass
    # to fully reunite -- a slower, safer trade-off than the alternative.
    n = len(clusters)
    original_members = [list(c["members"]) for c in clusters]
    locked: set[int] = set()  # can't take part in another merge check this pass
    absorbed: set[int] = set()  # merged into another cluster; dropped from the final result

    for a in range(n):
        if a in locked:
            continue
        for b in range(a + 1, n):
            if b in locked:
                continue
            result = strongest_inter_cluster_pair(original_members[a], original_members[b])
            if result is None:
                continue
            i, j, sim = result
            if sim < merge_candidate_min_sim:
                continue
            same = verify_fn(clusters[a]["crop_paths"][i], clusters[b]["crop_paths"][j])
            log.append(
                {
                    "type": "merge_check",
                    "a": clusters[a]["character_id"],
                    "b": clusters[b]["character_id"],
                    "sim": sim,
                    "same": same,
                }
            )
            if same:
                clusters[a]["members"] = original_members[a] + original_members[b]
                clusters[a]["crop_paths"] = clusters[a]["crop_paths"] + clusters[b]["crop_paths"]
                clusters[a]["count"] = len(clusters[a]["members"])
                locked.add(a)
                locked.add(b)
                absorbed.add(b)
                log.append(
                    {"type": "merge_applied", "into": clusters[a]["character_id"], "from": clusters[b]["character_id"]}
                )
                break  # `a` is now locked; stop checking it against further clusters this pass

    final_clusters = [c for idx, c in enumerate(clusters) if idx not in absorbed]
    return final_clusters, log
