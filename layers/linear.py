"""
layers/linear.py
-----------------
A fully-connected linear layer implemented with an explicit, manually
derived backward pass. No autograd is used anywhere.

Forward:
    Y = X @ W + b

Shapes:
    X: (..., in_features)
    W: (in_features, out_features)
    b: (out_features,)
    Y: (..., out_features)

Backward (given dY = dL/dY):
    dL/dW = X^T @ dY            (summed over all leading/batch dims)
    dL/db = sum(dY, over all leading/batch dims)
    dL/dX = dY @ W^T
"""

import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.02,
                 rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.W = (rng.standard_normal((in_features, out_features)) * init_scale).astype(np.float64)
        self.b = np.zeros((out_features,), dtype=np.float64)

        # Gradients (allocated lazily by zero_grad / backward)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backward
        self._x_shape = None
        self._x_flat = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Flatten all leading dims into a single "batch" dim for the matmul,
        # then reshape back. This lets the same code handle (B, T, D) or (N, D).
        self._x_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        self._x_flat = x_flat
        y_flat = x_flat @ self.W + self.b
        return y_flat.reshape(*x.shape[:-1], self.out_features)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dy_flat = dy.reshape(-1, self.out_features)

        # dL/dW = X^T @ dY   (sum over the batch dimension is implicit in matmul)
        self.dW = self._x_flat.T @ dy_flat
        # dL/db = sum over batch dimension
        self.db = dy_flat.sum(axis=0)
        # dL/dX = dY @ W^T
        dx_flat = dy_flat @ self.W.T

        return dx_flat.reshape(self._x_shape)

    def parameters(self):
        return {"W": self.W, "b": self.b}

    def gradients(self):
        return {"W": self.dW, "b": self.db}
