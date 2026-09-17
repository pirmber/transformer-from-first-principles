"""
tests/test_gradients.py
--------------------------
Runs numerical gradient checking (finite differences) across the key
components of the project and prints a summary report of:

    analytical gradient vs numerical gradient, absolute error, relative error

This file is meant to be run directly (`python tests/test_gradients.py`)
for a human-readable audit, as well as under pytest.
"""

import numpy as np

from layers.linear import Linear
from layers.layernorm import LayerNorm
from layers.gelu import GELU
from layers.softmax import softmax_forward, softmax_backward
from attention.multi_head_attention import MultiHeadAttention
from transformer.transformer_block import TransformerBlock
from tests.grad_check_utils import numerical_gradient, report


def check_linear():
    rng = np.random.default_rng(10)
    layer = Linear(4, 3, rng=rng)
    x = rng.standard_normal((2, 4))
    dy = rng.standard_normal((2, 3))
    layer.forward(x)
    dx = layer.backward(dy)
    num_dx = numerical_gradient(lambda x_: float(np.sum(layer.forward(x_) * dy)), x.copy())
    return report("Linear dX", dx, num_dx)


def check_layernorm():
    rng = np.random.default_rng(11)
    ln = LayerNorm(6)
    x = rng.standard_normal((2, 6))
    dy = rng.standard_normal((2, 6))
    ln.forward(x)
    dx = ln.backward(dy)
    num_dx = numerical_gradient(lambda x_: float(np.sum(ln.forward(x_) * dy)), x.copy())
    return report("LayerNorm dX", dx, num_dx)


def check_softmax():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((3, 5))
    dy = rng.standard_normal(x.shape)
    s = softmax_forward(x)
    dx = softmax_backward(dy, s)
    num_dx = numerical_gradient(lambda x_: float(np.sum(softmax_forward(x_) * dy)), x.copy())
    return report("Softmax dX", dx, num_dx)


def check_gelu():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((3, 5))
    dy = rng.standard_normal(x.shape)
    act = GELU()
    act.forward(x)
    dx = act.backward(dy)
    num_dx = numerical_gradient(lambda x_: float(np.sum(GELU().forward(x_) * dy)), x.copy())
    return report("GELU dX", dx, num_dx)


def check_attention():
    rng = np.random.default_rng(14)
    B, T, D, H = 2, 4, 8, 2
    attn = MultiHeadAttention(D, H, rng=rng)
    x = rng.standard_normal((B, T, D))
    dy = rng.standard_normal((B, T, D))
    attn.forward(x)
    dx = attn.backward(dy)
    num_dx = numerical_gradient(lambda x_: float(np.sum(attn.forward(x_) * dy)), x.copy(), eps=1e-5)
    return report("MultiHeadAttention dX", dx, num_dx, tol=1e-2)


def check_transformer_block():
    rng = np.random.default_rng(15)
    B, T, D, H, F = 2, 4, 8, 2, 16
    block = TransformerBlock(D, H, F, rng=rng)
    x = rng.standard_normal((B, T, D))
    dy = rng.standard_normal((B, T, D))
    block.forward(x)
    dx = block.backward(dy)
    num_dx = numerical_gradient(lambda x_: float(np.sum(block.forward(x_) * dy)), x.copy(), eps=1e-5)
    return report("TransformerBlock dX", dx, num_dx, tol=1e-2)


def test_gradient_check():
    results = [
        check_linear(),
        check_layernorm(),
        check_softmax(),
        check_gelu(),
        check_attention(),
        check_transformer_block(),
    ]
    assert all(results), "one or more gradient checks failed"


if __name__ == "__main__":
    test_gradient_check()
    print("\nAll gradient checks passed.")
