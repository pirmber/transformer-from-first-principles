import numpy as np
from layers.layernorm import LayerNorm
from tests.grad_check_utils import numerical_gradient, report


def test_layernorm_forward_backward():
    rng = np.random.default_rng(1)
    D = 6
    ln = LayerNorm(D)
    x = rng.standard_normal((2, 3, D))
    dy = rng.standard_normal((2, 3, D))

    y = ln.forward(x)
    assert y.shape == x.shape
    # Check normalization actually happened (mean ~0, var ~1 before affine)
    xhat = ln._xhat
    assert np.allclose(xhat.mean(axis=-1), 0, atol=1e-6)
    assert np.allclose(xhat.std(axis=-1), 1, atol=1e-3)

    dx = ln.backward(dy)

    def loss_of_x(x_):
        return float(np.sum(ln.forward(x_) * dy))

    num_dx = numerical_gradient(loss_of_x, x.copy())
    assert report("layernorm dX", dx, num_dx)

    gamma_orig = ln.gamma.copy()

    def loss_of_gamma(g):
        ln.gamma = g
        return float(np.sum(ln.forward(x) * dy))

    num_dgamma = numerical_gradient(loss_of_gamma, ln.gamma.copy())
    ln.gamma = gamma_orig
    ln.forward(x)
    ln.backward(dy)
    assert report("layernorm dgamma", ln.dgamma, num_dgamma)

    beta_orig = ln.beta.copy()

    def loss_of_beta(b):
        ln.beta = b
        return float(np.sum(ln.forward(x) * dy))

    num_dbeta = numerical_gradient(loss_of_beta, ln.beta.copy())
    ln.beta = beta_orig
    ln.forward(x)
    ln.backward(dy)
    assert report("layernorm dbeta", ln.dbeta, num_dbeta)


if __name__ == "__main__":
    test_layernorm_forward_backward()
    print("test_layernorm passed")
