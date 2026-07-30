"""
Visualization utilities for the Linear Probing Pipeline.

Generates layer-wise accuracy plots and selectivity gap charts.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from probing import ProbeResults

matplotlib.use("Agg")  # Non-interactive backend for server/headless environments


def plot_layer_accuracy(
    results: ProbeResults,
    attribute: str,
    classifier: str = "logistic",
    save_dir: str = "plots",
) -> str:
    """
    Plot probe accuracy vs. layer for real and control probes.

    Returns the path to the saved figure.
    """
    real = [
        r
        for r in results.results
        if r.attribute == attribute and r.classifier == classifier and not r.is_control
    ]
    ctrl = [
        r
        for r in results.results
        if r.attribute == attribute and r.classifier == classifier and r.is_control
    ]

    real = sorted(real, key=lambda r: r.layer)
    ctrl = sorted(ctrl, key=lambda r: r.layer)

    layers_r = [r.layer for r in real]
    acc_r = [r.accuracy for r in real]
    cv_r = [r.cv_std for r in real]

    layers_c = [r.layer for r in ctrl]
    acc_c = [r.accuracy for r in ctrl]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers_r, acc_r, "o-", label="Real probe", color="steelblue", linewidth=2)
    ax.fill_between(
        layers_r,
        np.array([a - s for a, s in zip(acc_r, cv_r)]),
        np.array([a + s for a, s in zip(acc_r, cv_r)]),
        alpha=0.15,
        color="steelblue",
    )
    ax.plot(layers_c, acc_c, "s--", label="Control (shuffled labels)", color="salmon", linewidth=2)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Linear Probe Accuracy — {attribute} ({classifier})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"probe_accuracy_{attribute}_{classifier}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved layer accuracy plot → {path}")
    return path


def plot_selectivity_gap(
    results: ProbeResults,
    attribute: str,
    classifier: str = "logistic",
    save_dir: str = "plots",
) -> str:
    """
    Bar chart showing the accuracy gap (real − control) per layer.
    """
    real = {
        r.layer: r.accuracy
        for r in results.results
        if r.attribute == attribute and r.classifier == classifier and not r.is_control
    }
    ctrl = {
        r.layer: r.accuracy
        for r in results.results
        if r.attribute == attribute and r.classifier == classifier and r.is_control
    }
    layers = sorted(real.keys())
    gaps = [real[layer] - ctrl.get(layer, 0) for layer in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["forestgreen" if g > 0 else "red" for g in gaps]
    ax.bar(layers, gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Selectivity Gap (Real − Control)", fontsize=12)
    ax.set_title(f"Selectivity Gap — {attribute}", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"selectivity_gap_{attribute}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved selectivity gap plot → {path}")
    return path
