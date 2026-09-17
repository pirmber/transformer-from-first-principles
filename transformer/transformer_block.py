"""
transformer/transformer_block.py
-----------------------------------
A single pre-LayerNorm Transformer block:

    x1 = x0 + MultiHeadAttention(LayerNorm1(x0))
    x2 = x1 + FeedForward(LayerNorm2(x1))

Residual connections (section 11 of the spec):
    For Y = X + F(X), backward is  dX = dY + dF/dX(dY)
    i.e. the incoming gradient dY flows unchanged down the identity/skip
    path AND is also passed into F's backward -- both contributions are
    added together. It is easy to accidentally forget the identity path;
    this implementation keeps it explicit below.
"""

import numpy as np

from layers.layernorm import LayerNorm
from attention.multi_head_attention import MultiHeadAttention
from transformer.feed_forward import FeedForward


class TransformerBlock:
    def __init__(self, embedding_dim: int, num_heads: int, ffn_dim: int,
                 eps: float = 1e-5, init_scale: float = 0.02,
                 rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        self.ln1 = LayerNorm(embedding_dim, eps)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, init_scale, rng)
        self.ln2 = LayerNorm(embedding_dim, eps)
        self.ffn = FeedForward(embedding_dim, ffn_dim, init_scale, rng)

    def forward(self, x0: np.ndarray) -> np.ndarray:
        ln1_out = self.ln1.forward(x0)
        attn_out = self.attn.forward(ln1_out)
        x1 = x0 + attn_out                       # residual connection 1

        ln2_out = self.ln2.forward(x1)
        ffn_out = self.ffn.forward(ln2_out)
        x2 = x1 + ffn_out                        # residual connection 2

        return x2

    def backward(self, dx2: np.ndarray) -> np.ndarray:
        # --- residual 2: x2 = x1 + ffn_out --------------------------------
        d_ffn_out = dx2
        dx1_identity = dx2                        # identity/skip path
        d_ln2_out = self.ffn.backward(d_ffn_out)
        dx1_through_ffn = self.ln2.backward(d_ln2_out)
        dx1 = dx1_identity + dx1_through_ffn      # sum both paths

        # --- residual 1: x1 = x0 + attn_out --------------------------------
        d_attn_out = dx1
        dx0_identity = dx1                        # identity/skip path
        d_ln1_out = self.attn.backward(d_attn_out)
        dx0_through_attn = self.ln1.backward(d_ln1_out)
        dx0 = dx0_identity + dx0_through_attn     # sum both paths

        return dx0

    def parameters(self):
        return {
            "ln1": self.ln1.parameters(), "attn": self.attn.parameters(),
            "ln2": self.ln2.parameters(), "ffn": self.ffn.parameters(),
        }

    def gradients(self):
        return {
            "ln1": self.ln1.gradients(), "attn": self.attn.gradients(),
            "ln2": self.ln2.gradients(), "ffn": self.ffn.gradients(),
        }

    def sub_layers(self):
        return [self.ln1, self.ln2] + self.attn.sub_layers() + self.ffn.sub_layers()
