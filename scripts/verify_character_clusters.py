"""Run the VLM verification pass (character_verification.py) on top of an
already-clustered comic's character store: checks each cluster's weakest
internal link (candidate false merge) and each cluster-pair's strongest
cross link (candidate missed merge) against Qwen2.5-VL, applies any
splits/merges it confirms, and reports before/after counts.

Requires `character_embedding` (the `dinov2` backend, by default) to have
already been run for this comic -- this script only refines its output, it
doesn't detect characters itself.

Usage:
    python scripts/verify_character_clusters.py <comic_id> [backend]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comicreel.character_store import load_store, save_store
from comicreel.character_verification import refine_clusters, verify_same_character

ROOT = Path(__file__).resolve().parent.parent
MODEL = "qwen2.5vl:7b"


def main(comic_id: str, backend_name: str = "dinov2") -> None:
    cache_root = ROOT / "data" / "cache"
    store = load_store(comic_id, cache_root, backend_name)
    before_count = len(store["clusters"])
    before_sizes = sorted((len(c["members"]) for c in store["clusters"]), reverse=True)

    def verify_fn(image_a: str, image_b: str) -> bool:
        same, _ = verify_same_character(image_a, image_b, MODEL)
        return same

    refined, log = refine_clusters(store["clusters"], verify_fn)
    store["clusters"] = refined
    save_store(comic_id, cache_root, backend_name, store)

    after_count = len(refined)
    after_sizes = sorted((len(c["members"]) for c in refined), reverse=True)

    splits = [e for e in log if e["type"] == "split_applied"]
    merges = [e for e in log if e["type"] == "merge_applied"]
    checks = [e for e in log if e["type"] in ("split_check", "merge_check")]

    print(f"=== {comic_id} ({backend_name}) ===")
    print(f"before: {before_count} clusters, sizes={before_sizes}")
    print(f"after:  {after_count} clusters, sizes={after_sizes}")
    print(f"VLM calls made: {len(checks)}  (splits applied: {len(splits)}, merges applied: {len(merges)})")
    for entry in log:
        print(f"  {entry}")

    out_dir = ROOT / f"{comic_id} output" / "phase4_character_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verification_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    comic_id = sys.argv[1]
    backend_name = sys.argv[2] if len(sys.argv) > 2 else "dinov2"
    main(comic_id, backend_name)
