"""
training/dataset.py
----------------------
Minimal dataset helper for a character-level causal language model.

Given a full token-id sequence, a training example is a contiguous window
of length `context_length`:

    input:  tokens[i : i+T]
    target: tokens[i+1 : i+T+1]     (i.e. "predict the next character")
"""

import numpy as np


class CharDataset:
    def __init__(self, token_ids: list, context_length: int):
        self.data = np.array(token_ids, dtype=np.int64)
        self.context_length = context_length
        assert len(self.data) > context_length + 1, "corpus too short for this context_length"

    def get_batch(self, batch_size: int, rng: np.random.Generator):
        T = self.context_length
        max_start = len(self.data) - T - 1
        starts = rng.integers(0, max_start, size=batch_size)

        x = np.stack([self.data[s:s + T] for s in starts])
        y = np.stack([self.data[s + 1:s + T + 1] for s in starts])
        return x, y
