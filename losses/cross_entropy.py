"""
losses/cross_entropy.py
--------------------------
Causal language-model cross-entropy loss.

Given logits of shape (B, T, V) and integer targets of shape (B, T)
(target[b, t] is the correct next token for position t):

    p = softmax(logits, axis=-1)
    loss = -mean( log(p[b, t, target[b, t]]) )   over all B*T positions

Gradient (the classic softmax + cross-entropy simplification):

    dLoss/dLogits = (p - one_hot(target)) / (B * T)

This simplification comes from differentiating -log(softmax(x)_y) w.r.t.
x, which collapses the softmax Jacobian and the cross-entropy gradient
into the very simple "probabilities minus one-hot" expression.
"""

import numpy as np

from layers.softmax import softmax_forward


class CrossEntropyLoss:
    def __init__(self):
        self._probs = None
        self._targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        B, T, V = logits.shape
        probs = softmax_forward(logits, axis=-1)

        # Numerical safety: clip away from exactly 0 before taking log.
        eps = 1e-12
        target_probs = np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0]
        target_probs = np.clip(target_probs, eps, 1.0)
        loss = -np.mean(np.log(target_probs))

        self._probs = probs
        self._targets = targets
        return float(loss)

    def backward(self) -> np.ndarray:
        probs = self._probs
        targets = self._targets
        B, T, V = probs.shape

        one_hot = np.zeros_like(probs)
        np.put_along_axis(one_hot, targets[..., None], 1.0, axis=-1)

        dlogits = (probs - one_hot) / (B * T)
        return dlogits
