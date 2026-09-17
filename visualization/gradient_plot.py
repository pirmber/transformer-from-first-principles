"""
visualization/gradient_plot.py
----------------------------------
Plots gradient-magnitude curves per component over training, and bar
charts of parameter statistics (mean / std / min / max) for selected
layers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_gradient_norms(grad_norm_by_component: dict, out_path: str = "gradient_norms.png",
                         components=None):
    """
    grad_norm_by_component: {name: [norm_step_1, norm_step_2, ...]}
    components: optional list of component-name substrings to include
                (e.g. ["token_embedding", "attn.Wq", "ln1", "ffn.fc1", "final_linear"])
    """
    if components is None:
        # Pick one representative component per requested category.
        keys = list(grad_norm_by_component.keys())
        wanted_substrings = ["token_embedding", "attn.Wq", "attn.Wo", "ln1", "ffn.fc1", "final_linear"]
        components = []
        for sub in wanted_substrings:
            for k in keys:
                if sub in k and k not in components:
                    components.append(k)
                    break

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in components:
        if name in grad_norm_by_component:
            ax.plot(grad_norm_by_component[name], label=name, linewidth=1.2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient L2 norm")
    ax.set_title("Gradient magnitude by component")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_parameter_stats(named_layers, out_path: str = "parameter_stats.png"):
    """named_layers: list of (name, layer) with layer.parameters() -> dict of arrays."""
    names, means, stds, mins, maxs = [], [], [], [], []
    for name, layer in named_layers:
        for pname, p in layer.parameters().items():
            names.append(f"{name}.{pname}")
            means.append(float(np.mean(p)))
            stds.append(float(np.std(p)))
            mins.append(float(np.min(p)))
            maxs.append(float(np.max(p)))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.errorbar(x, means, yerr=stds, fmt="o", color="#3b6fd6", label="mean +/- std", capsize=3)
    ax.scatter(x, mins, marker="_", color="red", label="min")
    ax.scatter(x, maxs, marker="_", color="green", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("Parameter value")
    ax.set_title("Parameter statistics by layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
