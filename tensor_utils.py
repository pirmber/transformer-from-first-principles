"""
tensor_utils.py
---------------
Small shared helpers used across the project. Kept intentionally tiny:
this project favors explicit math in each layer over hidden abstractions.
"""

import numpy as np


def assert_finite(x: np.ndarray, name: str = "tensor") -> None:
    """Raise if a tensor contains NaN/Inf. Used liberally for debugging."""
    if not np.all(np.isfinite(x)):
        raise FloatingPointError(f"{name} contains NaN or Inf values")


def global_grad_norm(grad_arrays) -> float:
    """L2 norm of a collection of gradient arrays, flattened and concatenated."""
    total = 0.0
    for g in grad_arrays:
        total += float(np.sum(g.astype(np.float64) ** 2))
    return float(np.sqrt(total))


def clip_grad_norm_(grad_arrays, max_norm: float) -> float:
    """
    Rescale a collection of gradient arrays *in place* so that their global
    L2 norm does not exceed max_norm. Returns the norm before clipping.
    """
    norm = global_grad_norm(grad_arrays)
    if norm > max_norm and norm > 0:
        scale = max_norm / (norm + 1e-6)
        for g in grad_arrays:
            g *= scale
    return norm
