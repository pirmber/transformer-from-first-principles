"""
visualization/attention_plot.py
----------------------------------
Renders an attention-weight heatmap for a single head, demonstrating the
causal mask: the strictly upper-right triangle (future tokens) should be
essentially zero.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_attention(attn_matrix: np.ndarray, tokens=None, out_path: str = "attention.png",
                    title: str = "Attention weights (causal)"):
    """attn_matrix: (T, T) numpy array of attention weights for one head/example."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(attn_matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key position (token attended to)")
    ax.set_ylabel("Query position (current token)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if tokens is not None and len(tokens) <= 40:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
