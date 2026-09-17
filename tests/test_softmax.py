import numpy as np
from layers.softmax import softmax_forward, softmax_backward, softmax_backward_jacobian
from layers.gelu import GELU
from tests.grad_check_utils import numerical_gradient, report


def test_softmax_forward_backward():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 5)) * 3.0
    dy = rng.standard_normal(x.shape)

    s = softmax_forward(x, axis=-1)
    assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-8)

    dx = softmax_backward(dy, s, axis=-1)

    def loss_of_x(x_):
        s_ = softmax_forward(x_, axis=-1)
        return float(np.sum(s_ * dy))

    num_dx = numerical_gradient(loss_of_x, x.copy())
    assert report("softmax dX (VJP)", dx, num_dx)

    # Cross-check the VJP formulation against the explicit Jacobian
    # formulation for a single row.
    row_x = x[0]
    row_dy = dy[0]
    row_s = softmax_forward(row_x, axis=-1)
    dx_jacobian = softmax_backward_jacobian(row_dy, row_s)
    dx_vjp = softmax_backward(row_dy, row_s, axis=-1)
    assert report("softmax jacobian vs VJP", dx_jacobian, dx_vjp)


def test_gelu_forward_backward():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 6))
    dy = rng.standard_normal(x.shape)

    act = GELU()
    y = act.forward(x)
    dx = act.backward(dy)

    def loss_of_x(x_):
        a = GELU()
        return float(np.sum(a.forward(x_) * dy))

    num_dx = numerical_gradient(loss_of_x, x.copy())
    assert report("gelu dX", dx, num_dx)


if __name__ == "__main__":
    test_softmax_forward_backward()
    test_gelu_forward_backward()
    print("test_softmax passed")
