"""Device auto-detection for CPU/CUDA/MPS backends."""

from __future__ import annotations


def resolve_device(requested: str) -> str:
    """Resolve "auto" to the best available device; pass through explicit choices."""
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
