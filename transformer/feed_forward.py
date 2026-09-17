"""
transformer/feed_forward.py
-----------------------------
Position-wise feed-forward network:

    FFN(X) = GELU(X @ W1 + b1) @ W2 + b2

Backward simply chains the (already-implemented) Linear and GELU
backward passes:

    dHidden      = Linear2.backward(dOut)
    dPreAct      = GELU.backward(dHidden)
    dX           = Linear1.backward(dPreAct)
"""

import numpy as np

from layers.linear import Linear
from layers.gelu import GELU


class FeedForward:
    def __init__(self, embedding_dim: int, ffn_dim: int, init_scale: float = 0.02,
                 rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        self.fc1 = Linear(embedding_dim, ffn_dim, init_scale, rng)
        self.act = GELU()
        self.fc2 = Linear(ffn_dim, embedding_dim, init_scale, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.fc1.forward(x)
        a = self.act.forward(h)
        y = self.fc2.forward(a)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        da = self.fc2.backward(dy)
        dh = self.act.backward(da)
        dx = self.fc1.backward(dh)
        return dx

    def parameters(self):
        return {"fc1": self.fc1.parameters(), "fc2": self.fc2.parameters()}

    def gradients(self):
        return {"fc1": self.fc1.gradients(), "fc2": self.fc2.gradients()}

    def sub_layers(self):
        return [self.fc1, self.fc2]
