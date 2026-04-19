"""Shared utilities for the PA-2 pipeline."""
import torch


def get_device(preferred: str = "cpu") -> torch.device:
    """
    Auto-detect best available device.
    Priority: CUDA → MPS (Apple Silicon) → CPU.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in ("cuda", "mps") and hasattr(torch.backends, "mps") \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
