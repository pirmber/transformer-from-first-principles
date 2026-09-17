"""
layers/gelu.py
---------------
GELU activation using the tanh approximation popularized by GPT-2:

    GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

Let:
    c = sqrt(2/pi)
    u = c * (x + 0.044715 * x^3)
    t = tanh(u)

Then GELU(x) = 0.5 * x * (1 + t).

Derivative (exact derivative of THIS approximation, not of the erf-based
exact GELU):

    d/dx GELU(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * du/dx

    du/dx = c * (1 + 3 * 0.044715 * x^2)

So:
    dGELU/dx = 0.5*(1+t) + 0.5*x*(1 - t^2) * c * (1 + 0.134145*x^2)
"""

import numpy as np

_C = np.sqrt(2.0 / np.pi)
_A = 0.044715


def gelu_forward(x: np.ndarray):
    u = _C * (x + _A * x ** 3)
    t = np.tanh(u)
    y = 0.5 * x * (1.0 + t)
    cache = (x, t)
    return y, cache


def gelu_backward(dy: np.ndarray, cache) -> np.ndarray:
    x, t = cache
    du_dx = _C * (1.0 + 3.0 * _A * x ** 2)
    dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du_dx
    return dy * dgelu_dx


class GELU:
    """Thin stateful wrapper matching the forward/backward layer interface."""

    def __init__(self):
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, cache = gelu_forward(x)
        self._cache = cache
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return gelu_backward(dy, self._cache)

    def parameters(self):
        return {}

    def gradients(self):
        return {}
