"""
layers/embedding.py
--------------------
Token and position embedding tables, implemented as simple lookup tables
(matrix rows selected by integer index).

Forward:
    X = token_embedding[token_ids] + position_embedding[position_ids]

Backward:
    Because embedding lookup just *selects* rows, the gradient for a given
    row is the sum of dOut over every location that used that row. The same
    token id can appear multiple times in a batch/sequence, so gradients
    must be accumulated (not overwritten) into repeated rows. We use
    np.add.at for exactly this reason (plain fancy-index assignment would
    silently drop duplicate contributions).
"""

import numpy as np


class TokenEmbedding:
    def __init__(self, vocab_size: int, embedding_dim: int, init_scale: float = 0.02,
                 rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        self.table = (rng.standard_normal((vocab_size, embedding_dim)) * init_scale).astype(np.float64)
        self.dtable = np.zeros_like(self.table)
        self._ids = None
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        self._ids = token_ids
        return self.table[token_ids]

    def backward(self, dy: np.ndarray) -> None:
        self.dtable = np.zeros_like(self.table)
        # Accumulate gradients into (possibly repeated) rows.
        np.add.at(self.dtable, self._ids, dy)

    def parameters(self):
        return {"table": self.table}

    def gradients(self):
        return {"table": self.dtable}


class PositionEmbedding:
    def __init__(self, max_len: int, embedding_dim: int, init_scale: float = 0.02,
                 rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        self.table = (rng.standard_normal((max_len, embedding_dim)) * init_scale).astype(np.float64)
        self.dtable = np.zeros_like(self.table)
        self._t = None
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def forward(self, seq_len: int, batch_size: int) -> np.ndarray:
        self._t = seq_len
        pos = self.table[:seq_len]              # (T, D)
        return np.broadcast_to(pos, (batch_size, seq_len, self.embedding_dim))

    def backward(self, dy: np.ndarray) -> None:
        # dy: (B, T, D). Sum over the batch dimension since the same
        # position rows were broadcast to every example in the batch.
        self.dtable = np.zeros_like(self.table)
        self.dtable[:self._t] = dy.sum(axis=0)

    def parameters(self):
        return {"table": self.table}

    def gradients(self):
        return {"table": self.dtable}
