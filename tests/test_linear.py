import numpy as np
from layers.linear import Linear
from tests.grad_check_utils import numerical_gradient, report


def test_linear_forward_backward():
    rng = np.random.default_rng(0)
    layer = Linear(5, 3, rng=rng)
    x = rng.standard_normal((2, 4, 5))

    y = layer.forward(x)
    assert y.shape == (2, 4, 3)

    dy = rng.standard_normal(y.shape)
    dx = layer.backward(dy)
    assert dx.shape == x.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape

    # Gradient check against a scalar loss = sum(dy * y)
    def loss_of_x(x_):
        return float(np.sum(layer.forward(x_) * dy))

    x_check = x.copy()
    num_dx = numerical_gradient(loss_of_x, x_check)
    assert report("linear dX", dx, num_dx)

    # Finite-diff check for W and b (numerical_gradient perturbs a single
    # array argument, so W and b are checked directly like this).
    W_orig = layer.W.copy()

    def loss_given_W(W):
        layer.W = W
        out = layer.forward(x)
        return float(np.sum(out * dy))

    num_dW = numerical_gradient(loss_given_W, layer.W.copy())
    layer.W = W_orig
    layer.forward(x)
    layer.backward(dy)
    assert report("linear dW", layer.dW, num_dW)

    b_orig = layer.b.copy()

    def loss_given_b(b):
        layer.b = b
        out = layer.forward(x)
        return float(np.sum(out * dy))

    num_db = numerical_gradient(loss_given_b, layer.b.copy())
    layer.b = b_orig
    layer.forward(x)
    layer.backward(dy)
    assert report("linear db", layer.db, num_db)


if __name__ == "__main__":
    test_linear_forward_backward()
    print("test_linear passed")
