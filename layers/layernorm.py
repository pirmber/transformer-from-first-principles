"""
layers/layernorm.py
--------------------
LayerNorm implemented with a fully analytical backward pass (no numerical
approximation).

Forward (normalizing over the last axis, dimension D):
    mu    = mean(x, axis=-1)
    var   = mean((x - mu)^2, axis=-1)
    xhat  = (x - mu) / sqrt(var + eps)
    y     = gamma * xhat + beta

Backward. Let N = D (the size of the normalized axis). Given dy = dL/dy:

    dgamma = sum(dy * xhat)                      summed over all non-feature dims
    dbeta  = sum(dy)                             summed over all non-feature dims

    dxhat  = dy * gamma

    # Standard LayerNorm gradient w.r.t. the input, derived from the chain
    # rule through mu and var (both of which depend on every element of x):
    dx = (1 / (N * std)) * (
            N * dxhat
            - sum(dxhat, axis=-1, keepdims=True)
            - xhat * sum(dxhat * xhat, axis=-1, keepdims=True)
         )

    where std = sqrt(var + eps).

This is the standard closed-form LayerNorm backward formula, derived by
differentiating y w.r.t. x through both mu(x) and var(x).
"""

import numpy as np


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones((dim,), dtype=np.float64)
        self.beta = np.zeros((dim,), dtype=np.float64)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        # cache
        self._xhat = None
        self._std = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        xhat = (x - mu) / std

        self._xhat = xhat
        self._std = std

        return self.gamma * xhat + self.beta

    def backward(self, dy: np.ndarray) -> np.ndarray:
        xhat = self._xhat
        std = self._std
        N = self.dim

        # Sum gradients for gamma/beta over every axis except the last
        # (i.e. over batch and sequence position).
        reduce_axes = tuple(range(dy.ndim - 1))
        self.dgamma = (dy * xhat).sum(axis=reduce_axes)
        self.dbeta = dy.sum(axis=reduce_axes)

        dxhat = dy * self.gamma

        sum_dxhat = dxhat.sum(axis=-1, keepdims=True)
        sum_dxhat_xhat = (dxhat * xhat).sum(axis=-1, keepdims=True)

        dx = (1.0 / (N * std)) * (
            N * dxhat - sum_dxhat - xhat * sum_dxhat_xhat
        )
        return dx

    def parameters(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def gradients(self):
        return {"gamma": self.dgamma, "beta": self.dbeta}
