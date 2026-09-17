"""
layers/softmax.py
------------------
Numerically stable softmax with two backward formulations:

1. `softmax_backward_jacobian` -- the general, textbook Jacobian formulation.
   For a single probability vector s of length K:
       dS_i/dx_j = s_i * (delta_ij - s_j)
   so the full Jacobian J is a (K, K) matrix, and dx = J @ dy.
   This is O(K^2) per row and mostly useful for teaching / verification.

2. `softmax_backward` -- the efficient vector-Jacobian product (VJP), which
   avoids ever forming the Jacobian explicitly:
       dx = s * (dy - sum(dy * s, axis=-1, keepdims=True))
   This is O(K) per row (same asymptotic cost as the forward pass) and is
   what any real training loop should use. It is mathematically identical
   to `s * (dy @ (I - s s^T))` but avoids the O(K^2) matrix multiply.

The efficient version is preferable during training because attention
matrices can have thousands of softmax rows (batch x heads x query
positions), each over a row of size T (key positions). Forming a (T, T)
Jacobian per row would multiply the compute and memory cost by T for no
benefit, since the VJP form produces the exact same gradient.
"""

import numpy as np


def softmax_forward(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: subtract the row max before exponentiating."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(dy: np.ndarray, s: np.ndarray, axis: int = -1) -> np.ndarray:
    """Efficient vector-Jacobian product. `s` is the softmax output from forward."""
    dot = np.sum(dy * s, axis=axis, keepdims=True)
    return s * (dy - dot)


def softmax_backward_jacobian(dy: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    General Jacobian formulation, provided for education/verification only.
    Only supports a single 1-D probability vector `s` of shape (K,) with
    dy of shape (K,).
    """
    assert s.ndim == 1 and dy.ndim == 1, "jacobian version only supports 1D vectors"
    K = s.shape[0]
    jacobian = np.diag(s) - np.outer(s, s)   # (K, K): dS_i/dx_j
    return jacobian @ dy
