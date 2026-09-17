"""
train.py
---------
End-to-end entrypoint:

  1. Load a small corpus.
  2. Build the character tokenizer.
  3. Construct the Transformer.
  4. Run gradient checks (sanity check before spending time training).
  5. Train the model, printing progress.
  6. Save model parameters.
  7. Generate sample text.
  8. Produce training / attention / gradient visualizations.

Run with:
    python train.py
"""

import os
import pickle

import numpy as np

from config import TransformerConfig
from tokenizer import CharTokenizer
from transformer.transformer import Transformer
from training.dataset import CharDataset
from training.trainer import Trainer
from generation.generate import generate
from visualization.loss_plot import plot_loss
from visualization.attention_plot import plot_attention
from visualization.gradient_plot import plot_gradient_norms, plot_parameter_stats

# A small, original, self-contained corpus. Repeated so that the tiny
# char-level model has enough examples of its (simple, repetitive)
# patterns to actually learn something within a few hundred steps.
CORPUS_PARAGRAPH = (
    "the quick brown fox jumps over the lazy dog. "
    "a transformer learns by passing gradients backward through every layer. "
    "attention lets each token look at the tokens that came before it. "
    "the cat sat on the mat and watched the rain fall softly outside. "
    "numbers and letters and words all become tokens in the end. "
)
CORPUS = CORPUS_PARAGRAPH * 40


def run_gradient_checks():
    print("=" * 70)
    print("STEP 1: Gradient checking (finite differences vs analytical grads)")
    print("=" * 70)
    # Imported here so a failure in gradient checking is loud and immediate.
    from tests.test_gradients import test_gradient_check
    test_gradient_check()
    print("All gradient checks passed.\n")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    run_gradient_checks()

    print("=" * 70)
    print("STEP 2: Building tokenizer and dataset")
    print("=" * 70)
    tokenizer = CharTokenizer(CORPUS)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=48,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        batch_size=16,
        learning_rate=3e-3,
        num_steps=600,
        seed=1337,
    )
    print(config)

    token_ids = tokenizer.encode(CORPUS)
    dataset = CharDataset(token_ids, config.context_length)

    print("\n" + "=" * 70)
    print("STEP 3: Constructing the Transformer")
    print("=" * 70)
    rng = np.random.default_rng(config.seed)
    model = Transformer(config, rng)
    n_params = sum(p.size for layer in model.all_sub_layers() for p in layer.parameters().values())
    print(f"Total trainable parameters: {n_params:,}")

    print("\n" + "=" * 70)
    print("STEP 4: Training")
    print("=" * 70)
    trainer = Trainer(model, dataset, config)
    history = trainer.train(num_steps=config.num_steps, log_every=50)

    print("\n" + "=" * 70)
    print("STEP 5: Saving model parameters")
    print("=" * 70)
    params_path = os.path.join(out_dir, "model_params.pkl")
    all_params = {name: {k: v.copy() for k, v in layer.parameters().items()}
                  for name, layer in model.named_sub_layers()}
    with open(params_path, "wb") as f:
        pickle.dump({"config": config, "params": all_params, "vocab": tokenizer.itos}, f)
    print(f"Saved parameters to {params_path}")

    print("\n" + "=" * 70)
    print("STEP 6: Generating sample text")
    print("=" * 70)
    sample = generate(model, tokenizer, prompt="the ", max_new_tokens=200,
                       temperature=0.8, top_k=10, rng=np.random.default_rng(0))
    print(sample)
    with open(os.path.join(out_dir, "sample_generation.txt"), "w") as f:
        f.write(sample)

    print("\n" + "=" * 70)
    print("STEP 7: Producing visualizations")
    print("=" * 70)
    plot_loss(history, out_path=os.path.join(out_dir, "loss.png"))

    # Grab an attention matrix from the first head of the first block for a
    # fresh forward pass over a real chunk of the corpus.
    x_vis, _ = dataset.get_batch(1, np.random.default_rng(123))
    model.forward(x_vis)
    attn_weights = model.blocks[0].attn._cache["attn"][0, 0]  # (T,T)
    tokens = [tokenizer.itos[i] for i in x_vis[0]]
    plot_attention(attn_weights, tokens=tokens, out_path=os.path.join(out_dir, "attention.png"))

    plot_gradient_norms(trainer.grad_norm_by_component, out_path=os.path.join(out_dir, "gradient_norms.png"))
    plot_parameter_stats(model.named_sub_layers(), out_path=os.path.join(out_dir, "parameter_stats.png"))

    print(f"Visualizations saved to {out_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
