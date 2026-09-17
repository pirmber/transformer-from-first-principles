"""
attention/causal_mask.py
-------------------------
Builds the additive causal mask used before softmax in self-attention.

For a sequence of length T, position i (query) may only attend to
positions j <= i (keys). We express this as an additive mask of shape
(T, T):

    mask[i, j] = 0            if j <= i   (visible)
    mask[i, j] = -1e10        if j >  i   (future, hidden)

Adding this to the raw attention scores before softmax drives the
probability of attending to future positions to (numerically) zero,
without ever producing NaN/Inf (we deliberately use a large finite
negative number rather than -inf, so that exp() and later gradient
computations stay finite).

The mask is a constant with respect to the model's parameters: it has no
trainable entries, so backward() does not need to compute gradients for it.
Because scores at masked positions become ~0 after softmax, the softmax
backward formula (s * (dy - sum(dy*s))) naturally produces ~0 gradient at
those masked positions too -- no special-casing is required in
`softmax_backward`.
"""

import numpy as np

NEG_INF = -1e10


def build_causal_mask(seq_len: int) -> np.ndarray:
    """Returns an additive mask of shape (seq_len, seq_len)."""
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    mask = np.where(j > i, NEG_INF, 0.0)
    return mask
