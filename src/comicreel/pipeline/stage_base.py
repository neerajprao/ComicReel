"""Abstract base class every pipeline stage implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    name: str

    @abstractmethod
    def run(self, comic_id: str, page_id: str, prior_artifacts: dict[str, Any]) -> dict[str, Any]:
        """Run this stage for one page given prior stages' artifacts.

        Returns this stage's artifact as a plain dict (validated against the
        matching schema in artifacts/schemas.py by the caller).
        """
        raise NotImplementedError
