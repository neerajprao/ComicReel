"""Content-hash based cache for pipeline stage artifacts.

Keys each stage's output by a hash of (input bytes + stage config), so a
cached result is reused automatically whenever neither the input nor the
config for that stage has changed. Backs `--resume` and per-stage re-runs
without recomputing everything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class StageCache:
    def __init__(self, cache_root: Path | str):
        self.cache_root = Path(cache_root)

    def _path(self, comic_id: str, page_id: str, stage: str, input_hash: str) -> Path:
        return self.cache_root / comic_id / page_id / stage / f"{input_hash}.json"

    @staticmethod
    def hash_inputs(*, input_bytes: bytes = b"", config: dict[str, Any] | None = None) -> str:
        h = hashlib.sha256()
        h.update(input_bytes)
        if config:
            h.update(json.dumps(config, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def has(self, comic_id: str, page_id: str, stage: str, input_hash: str) -> bool:
        return self._path(comic_id, page_id, stage, input_hash).exists()

    def get(self, comic_id: str, page_id: str, stage: str, input_hash: str) -> dict[str, Any] | None:
        path = self._path(comic_id, page_id, stage, input_hash)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(
        self, comic_id: str, page_id: str, stage: str, input_hash: str, result: dict[str, Any]
    ) -> Path:
        path = self._path(comic_id, page_id, stage, input_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        return path
