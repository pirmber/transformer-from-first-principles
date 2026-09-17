import numpy as np
from transformer.transformer_block import TransformerBlock
from transformer.transformer import Transformer
from config import TransformerConfig
from tests.grad_check_utils import numerical_gradient, report


def test_transformer_block_forward_backward():
    rng = np.random.default_rng(5)
    B, T, D, H, F = 2, 4, 8, 2, 16
    block = TransformerBlock(D, H, F, rng=rng)
    x = rng.standard_normal((B, T, D))
    dy = rng.standard_normal((B, T, D))

    y = block.forward(x)
    assert y.shape == x.shape

    dx = block.backward(dy)
    assert dx.shape == x.shape

    def loss_of_x(x_):
        return float(np.sum(block.forward(x_) * dy))

    num_dx = numerical_gradient(loss_of_x, x.copy(), eps=1e-5)
    assert report("transformer_block dX", dx, num_dx, tol=1e-2)


def test_full_transformer_forward_backward_shapes():
    cfg = TransformerConfig(vocab_size=20, context_length=8, embedding_dim=16,
                             num_heads=4, num_layers=2, ffn_dim=32, seed=42)
    rng = np.random.default_rng(cfg.seed)
    model = Transformer(cfg, rng)

    B, T = 3, 6
    ids = rng.integers(0, cfg.vocab_size, size=(B, T))
    logits = model.forward(ids)
    assert logits.shape == (B, T, cfg.vocab_size)

    dlogits = rng.standard_normal(logits.shape) * 0.01
    model.backward(dlogits)

    # Every parameter should have a same-shaped, finite gradient.
    for name, layer in model.named_sub_layers():
        params = layer.parameters()
        grads = layer.gradients()
        for key in params:
            assert params[key].shape == grads[key].shape, f"{name}.{key} shape mismatch"
            assert np.all(np.isfinite(grads[key])), f"{name}.{key} grad has NaN/Inf"


def test_full_transformer_gradient_check():
    # Tiny model, tiny finite-difference check on the token embedding table,
    # using the real cross-entropy loss end to end.
    from losses.cross_entropy import CrossEntropyLoss

    cfg = TransformerConfig(vocab_size=8, context_length=5, embedding_dim=8,
                             num_heads=2, num_layers=1, ffn_dim=16, seed=7)
    rng = np.random.default_rng(cfg.seed)
    model = Transformer(cfg, rng)
    loss_fn = CrossEntropyLoss()

    B, T = 2, 5
    ids = rng.integers(0, cfg.vocab_size, size=(B, T))
    targets = rng.integers(0, cfg.vocab_size, size=(B, T))

    def full_loss():
        logits = model.forward(ids)
        return loss_fn.forward(logits, targets)

    loss = full_loss()
    dlogits = loss_fn.backward()
    model.backward(dlogits)
    analytical = model.token_emb.dtable.copy()

    def loss_given_table(table):
        model.token_emb.table = table
        return full_loss()

    num_grad = numerical_gradient(loss_given_table, model.token_emb.table.copy(), eps=1e-4)
    assert report("full model dTokenEmbedding", analytical, num_grad, tol=5e-2)


if __name__ == "__main__":
    test_transformer_block_forward_backward()
    test_full_transformer_forward_backward_shapes()
    test_full_transformer_gradient_check()
    print("test_transformer passed")
