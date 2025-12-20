"""Utilities for GPU."""

import torch


def get_device():
    """Return the best available torch device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
