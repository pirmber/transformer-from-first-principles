"""
transformer/transformer.py
-----------------------------
The full decoder-only, GPT-style Transformer, assembled from the layers
built elsewhere in this project. Nothing here uses automatic
differentiation -- forward() caches what backward() needs, and backward()
manually threads gradients through every component in reverse order.

Forward:
    X = TokenEmbedding[ids] + PositionEmbedding[positions]
    for block in blocks: X = block(X)
    X = LayerNorm_final(X)
    logits = FinalLinear(X)

Backward (section 14 of the spec):
    dLogits
      -> FinalLinear.backward
      -> LayerNorm_final.backward
      -> block[-1].backward -> ... -> block[0].backward
      -> split gradient into token-embedding and position-embedding grads
"""

import numpy as np

from layers.embedding import TokenEmbedding, PositionEmbedding
from layers.layernorm import LayerNorm
from layers.linear import Linear
from transformer.transformer_block import TransformerBlock


class Transformer:
    def __init__(self, config, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng(config.seed)
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.embedding_dim,
                                         config.init_scale, rng)
        self.pos_emb = PositionEmbedding(config.context_length, config.embedding_dim,
                                          config.init_scale, rng)

        self.blocks = [
            TransformerBlock(config.embedding_dim, config.num_heads, config.ffn_dim,
                              config.layernorm_eps, config.init_scale, rng)
            for _ in range(config.num_layers)
        ]

        self.ln_f = LayerNorm(config.embedding_dim, config.layernorm_eps)
        self.final_linear = Linear(config.embedding_dim, config.vocab_size,
                                    config.init_scale, rng)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (B, T) int array -> logits: (B, T, vocab_size)"""
        B, T = token_ids.shape
        assert T <= self.config.context_length, "sequence longer than context_length"

        tok = self.token_emb.forward(token_ids)          # (B,T,D)
        pos = self.pos_emb.forward(T, B)                  # (B,T,D)
        x = tok + pos

        for block in self.blocks:
            x = block.forward(x)

        x = self.ln_f.forward(x)
        logits = self.final_linear.forward(x)
        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        dx = self.final_linear.backward(dlogits)
        dx = self.ln_f.backward(dx)

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # x = tok + pos  ->  gradient flows unchanged to both embeddings
        self.token_emb.backward(dx)
        self.pos_emb.backward(dx)

    def all_sub_layers(self):
        """Every component exposing parameters()/gradients(), for the optimizer
        and for gradient-norm / parameter-statistics visualizations."""
        layers = [self.token_emb, self.pos_emb, self.ln_f, self.final_linear]
        for block in self.blocks:
            layers.extend(block.sub_layers())
        return layers

    def named_sub_layers(self):
        """Same as all_sub_layers but with human-readable names, for plots."""
        named = [
            ("token_embedding", self.token_emb),
            ("position_embedding", self.pos_emb),
        ]
        for i, block in enumerate(self.blocks):
            named.append((f"block{i}.ln1", block.ln1))
            named.append((f"block{i}.attn.Wq", block.attn.Wq))
            named.append((f"block{i}.attn.Wk", block.attn.Wk))
            named.append((f"block{i}.attn.Wv", block.attn.Wv))
            named.append((f"block{i}.attn.Wo", block.attn.Wo))
            named.append((f"block{i}.ln2", block.ln2))
            named.append((f"block{i}.ffn.fc1", block.ffn.fc1))
            named.append((f"block{i}.ffn.fc2", block.ffn.fc2))
        named.append(("ln_f", self.ln_f))
        named.append(("final_linear", self.final_linear))
        return named
