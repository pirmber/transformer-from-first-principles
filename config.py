"""
config.py
---------
Central configuration for the from-scratch Transformer.

All sizes are kept deliberately small so the whole pipeline
(forward pass, manual backward pass, gradient checking, training,
and generation) runs comfortably on a CPU in seconds.
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # --- Model size ---------------------------------------------------
    vocab_size: int = 65          # filled in from the tokenizer at runtime
    context_length: int = 64      # max sequence length (T)
    embedding_dim: int = 64       # model dimension (D)
    num_heads: int = 4            # number of attention heads (H)
    num_layers: int = 2           # number of Transformer blocks
    ffn_dim: int = 256            # hidden dimension of the feed-forward net

    # --- Numerics -------------------------------------------------------
    layernorm_eps: float = 1e-5
    init_scale: float = 0.02      # std-dev used for weight initialization

    # --- Training ---------------------------------------------------------
    batch_size: int = 16
    learning_rate: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    num_steps: int = 500
    seed: int = 1337

    @property
    def head_dim(self) -> int:
        assert self.embedding_dim % self.num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )
        return self.embedding_dim // self.num_heads
