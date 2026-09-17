import numpy as np
from attention.multi_head_attention import MultiHeadAttention
from attention.causal_mask import build_causal_mask
from tests.grad_check_utils import numerical_gradient, report


def test_causal_mask_shape_and_values():
    T = 4
    mask = build_causal_mask(T)
    # Visible (lower-triangular incl diagonal) entries must be 0.
    for i in range(T):
        for j in range(T):
            if j <= i:
                assert mask[i, j] == 0.0
            else:
                assert mask[i, j] < -1e5


def test_attention_forward_backward():
    rng = np.random.default_rng(4)
    B, T, D, H = 2, 4, 8, 2
    attn = MultiHeadAttention(D, H, rng=rng)
    x = rng.standard_normal((B, T, D))
    dy = rng.standard_normal((B, T, D))

    out = attn.forward(x)
    assert out.shape == (B, T, D)

    # Verify causal masking: attention weights to future positions ~ 0.
    weights = attn._cache["attn"]  # (B,H,T,T)
    future_mask = np.triu(np.ones((T, T)), k=1).astype(bool)
    assert np.all(weights[:, :, future_mask] < 1e-6)
    # Each row of attention weights sums to 1.
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    dx = attn.backward(dy)
    assert dx.shape == x.shape

    def loss_of_x(x_):
        return float(np.sum(attn.forward(x_) * dy))

    num_dx = numerical_gradient(loss_of_x, x.copy(), eps=1e-5)
    assert report("attention dX", dx, num_dx, tol=1e-2)

    # Spot-check a weight gradient (Wq.W) via finite differences.
    Wq = attn.Wq
    W_orig = Wq.W.copy()

    def loss_of_Wq(W):
        Wq.W = W
        return float(np.sum(attn.forward(x) * dy))

    num_dWq = numerical_gradient(loss_of_Wq, Wq.W.copy(), eps=1e-5)
    Wq.W = W_orig
    attn.forward(x)
    attn.backward(dy)
    assert report("attention dWq", Wq.dW, num_dWq, tol=1e-2)


if __name__ == "__main__":
    test_causal_mask_shape_and_values()
    test_attention_forward_backward()
    print("test_attention passed")
