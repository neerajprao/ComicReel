"""Importing this package registers every implemented stage backend
(via the `@register_backend` decorator) with `comicreel.config`."""

from comicreel.stages import (  # noqa: F401
    bubble_detection,
    character_embedding,
    panel_detection,
    vlm_extraction,
)
