"""
attention/multi_head_attention.py
-----------------------------------
Multi-head causal self-attention with a fully manual backward pass.

Forward (per head):
    Q = X @ WQ,   K = X @ WK,   V = X @ WV
    S = (Q @ K^T) / sqrt(d_k)              # raw scores
    S = S + causal_mask                    # hide future positions
    A = softmax(S, axis=-1)                # attention weights
    O = A @ V                              # weighted sum of values

Heads are computed in parallel by reshaping the embedding dimension D into
(H, d_k) and moving H next to the batch dimension, then concatenated back
and passed through an output projection WO.

Backward chain (see project spec section 10), reproduced here exactly:

    dOutput
       |
       v
    dA = dHeadOut @ V^T   ,   dV = A^T @ dHeadOut
       |
       v
    dS = softmax_backward(dA, A)
       |
       v
    dQ = dS @ K / sqrt(d_k)  ,  dK = dS^T @ Q / sqrt(d_k)

Then, because X feeds into Q, K, and V independently, the gradient
w.r.t. X is the SUM of the three contributions:

    dX = dX_from_Q + dX_from_K + dX_from_V
"""

import numpy as np

from layers.linear import Linear
from layers.softmax import softmax_forward, softmax_backward
from attention.causal_mask import build_causal_mask


class MultiHeadAttention:
    def __init__(self, embedding_dim: int, num_heads: int, init_scale: float = 0.02,
                 rng: np.random.Generator = None):
        assert embedding_dim % num_heads == 0
        rng = rng or np.random.default_rng()

        self.D = embedding_dim
        self.H = num_heads
        self.dk = embedding_dim // num_heads

        self.Wq = Linear(embedding_dim, embedding_dim, init_scale, rng)
        self.Wk = Linear(embedding_dim, embedding_dim, init_scale, rng)
        self.Wv = Linear(embedding_dim, embedding_dim, init_scale, rng)
        self.Wo = Linear(embedding_dim, embedding_dim, init_scale, rng)

        # cache
        self._cache = None
        self._mask_cache = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # (B, T, D) -> (B, H, T, dk)
        B, T, D = x.shape
        x = x.reshape(B, T, self.H, self.dk)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        # (B, H, T, dk) -> (B, T, D)
        B, H, T, dk = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(B, T, H * dk)

    def _get_mask(self, T: int) -> np.ndarray:
        if T not in self._mask_cache:
            self._mask_cache[T] = build_causal_mask(T)
        return self._mask_cache[T]

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape

        Q = self._split_heads(self.Wq.forward(x))   # (B,H,T,dk)
        K = self._split_heads(self.Wk.forward(x))
        V = self._split_heads(self.Wv.forward(x))

        scale = 1.0 / np.sqrt(self.dk)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale   # (B,H,T,T)
        scores = scores + self._get_mask(T)[None, None, :, :]

        attn = softmax_forward(scores, axis=-1)                   # (B,H,T,T)
        head_out = np.matmul(attn, V)                              # (B,H,T,dk)

        concat = self._merge_heads(head_out)                       # (B,T,D)
        out = self.Wo.forward(concat)

        self._cache = dict(Q=Q, K=K, V=V, attn=attn, scale=scale, T=T)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        cache = self._cache
        Q, K, V, attn, scale = cache["Q"], cache["K"], cache["V"], cache["attn"], cache["scale"]

        # Through output projection
        dconcat = self.Wo.backward(dout)                 # (B,T,D)
        d_head_out = self._split_heads(dconcat)           # (B,H,T,dk)

        # O = A @ V
        dV = np.matmul(attn.transpose(0, 1, 3, 2), d_head_out)      # (B,H,T,dk)
        dA = np.matmul(d_head_out, V.transpose(0, 1, 3, 2))         # (B,H,T,T)

        # Softmax backward (efficient vector-Jacobian product)
        dS = softmax_backward(dA, attn, axis=-1)                     # (B,H,T,T)
        # (masked positions have attn ~ 0, so dS there is already ~0
        #  automatically -- no separate masking of the gradient is needed.)

        # S = Q @ K^T * scale
        dQ = np.matmul(dS, K) * scale                                 # (B,H,T,dk)
        dK = np.matmul(dS.transpose(0, 1, 3, 2), Q) * scale            # (B,H,T,dk)

        # Merge heads back to (B,T,D) before going through the Q/K/V
        # projection layers' own backward passes.
        dQ_merged = self._merge_heads(dQ)
        dK_merged = self._merge_heads(dK)
        dV_merged = self._merge_heads(dV)

        dX_from_Q = self.Wq.backward(dQ_merged)
        dX_from_K = self.Wk.backward(dK_merged)
        dX_from_V = self.Wv.backward(dV_merged)

        # X feeds Q, K, and V independently -> gradients add up.
        dX = dX_from_Q + dX_from_K + dX_from_V
        return dX

    def parameters(self):
        return {
            "Wq": self.Wq.parameters(), "Wk": self.Wk.parameters(),
            "Wv": self.Wv.parameters(), "Wo": self.Wo.parameters(),
        }

    def gradients(self):
        return {
            "Wq": self.Wq.gradients(), "Wk": self.Wk.gradients(),
            "Wv": self.Wv.gradients(), "Wo": self.Wo.gradients(),
        }

    def sub_layers(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]
