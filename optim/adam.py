"""
optim/adam.py
--------------
Adam optimizer implemented from scratch (no torch.optim.Adam or similar).

For each parameter theta with gradient g at step t:

    m_t = beta1 * m_{t-1} + (1 - beta1) * g
    v_t = beta2 * v_{t-1} + (1 - beta2) * g^2

    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)

    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

The optimizer operates directly on the leaf layers of the model (objects
exposing `.parameters()` and `.gradients()`, each returning a flat dict of
same-shaped numpy arrays). Updates are performed in-place on the parameter
arrays (`param -= ...`), so the very same array objects referenced by the
layers are updated -- no separate "set parameter" step is required.
"""

import numpy as np


class Adam:
    def __init__(self, layers, lr: float = 3e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # state[(id(layer), key)] = (m, v)
        self.state = {}
        for layer in layers:
            for key, p in layer.parameters().items():
                self.state[(id(layer), key)] = (np.zeros_like(p), np.zeros_like(p))

    def step(self) -> None:
        self.t += 1
        b1, b2, eps, lr, t = self.beta1, self.beta2, self.eps, self.lr, self.t

        bias_correction1 = 1.0 - b1 ** t
        bias_correction2 = 1.0 - b2 ** t

        for layer in self.layers:
            params = layer.parameters()
            grads = layer.gradients()
            for key, p in params.items():
                g = grads[key]
                m, v = self.state[(id(layer), key)]

                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * (g * g)
                self.state[(id(layer), key)] = (m, v)

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                p -= update   # in-place update: mutates the layer's own array

    def zero_grad(self) -> None:
        for layer in self.layers:
            for key, g in layer.gradients().items():
                g.fill(0.0)
