"""visualization/loss_plot.py -- plots training loss vs. step."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(history: dict, out_path: str = "loss.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["step"], history["loss"], color="#3b6fd6", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
