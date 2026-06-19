"""
Extended analysis script for probing experiments.

Runs additional analyses on cached hidden states:
  1. Token position comparison (requires two extraction runs)
  2. Regularization sweep (vary C)
  3. Layer subset analysis (early / mid / late)
  4. Per-class analysis (precision, recall, confusion matrix per layer)

Usage:
    python analysis.py --hidden-states results/gender_subsample/hidden_states.npz \
                       --data-file data/dataset_gender.json --label-key gender \
                       --results-dir results/gender_analysis --plots-dir plots/gender_analysis

    # Compare token positions (need two hidden state files):
    python analysis.py --hidden-states results/gender_last/hidden_states.npz \
                       --hidden-states-alt results/gender_mean/hidden_states.npz \
                       --data-file data/dataset_gender.json --label-key gender
"""

from extraction import load_hidden_states
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# 1. Token position comparison
# ---------------------------------------------------------------------------

def compare_token_positions(
    hs_last: Dict[int, np.ndarray],
    hs_mean: Dict[int, np.ndarray],
    labels: List[str],
    label_key: str,
    C: float = 1.0,
    cv_folds: int = 5,
    seed: int = 42,
    save_dir: str = "plots",
) -> dict:
    """Compare probe accuracy between 'last' and 'mean' token positions."""
    le = LabelEncoder()
    y = le.fit_transform(labels)

    layers = sorted(set(hs_last.keys()) & set(hs_mean.keys()))
    results = {"layers": layers, "last_acc": [], "mean_acc": []}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for layer in layers:
        for hs, key in [(hs_last, "last_acc"), (hs_mean, "mean_acc")]:
            X = hs[layer]
            scores = []
            for train_idx, test_idx in cv.split(X, y):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
                clf.fit(X_train, y[train_idx])
                scores.append(accuracy_score(y[test_idx], clf.predict(X_test)))
            results[key].append(float(np.mean(scores)))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, results["last_acc"], "o-", label="Last token", color="steelblue", linewidth=2)
    ax.plot(layers, results["mean_acc"], "s-", label="Mean pooling", color="darkorange", linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("CV Accuracy", fontsize=12)
    ax.set_title(f"Token Position Comparison — {label_key}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"token_position_comparison_{label_key}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Token position comparison → {path}")

    return results


# ---------------------------------------------------------------------------
# 2. Regularization sweep
# ---------------------------------------------------------------------------

def regularization_sweep(
    hidden_states: Dict[int, np.ndarray],
    labels: List[str],
    label_key: str,
    C_values: List[float] = None,
    cv_folds: int = 5,
    seed: int = 42,
    save_dir: str = "plots",
) -> dict:
    """Sweep over regularization strengths C and report accuracy per layer."""
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    layers = sorted(hidden_states.keys())

    results = {"C_values": C_values, "layers": layers, "accuracies": {}}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for C in C_values:
        accs_per_layer = []
        for layer in layers:
            X = hidden_states[layer]
            scores = []
            for train_idx, test_idx in cv.split(X, y):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
                clf.fit(X_train, y[train_idx])
                scores.append(accuracy_score(y[test_idx], clf.predict(X_test)))
            accs_per_layer.append(float(np.mean(scores)))
        results["accuracies"][str(C)] = accs_per_layer

    # Plot: one line per C value
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(C_values)))
    for i, C in enumerate(C_values):
        ax.plot(layers, results["accuracies"][str(C)], "o-",
                label=f"C={C}", color=cmap[i], linewidth=1.5, markersize=4)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("CV Accuracy", fontsize=12)
    ax.set_title(f"Regularization Sweep — {label_key}", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"regularization_sweep_{label_key}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Regularization sweep → {path}")

    return results


# ---------------------------------------------------------------------------
# 3. Layer subset analysis
# ---------------------------------------------------------------------------

def layer_subset_analysis(
    hidden_states: Dict[int, np.ndarray],
    labels: List[str],
    label_key: str,
    C: float = 1.0,
    cv_folds: int = 5,
    seed: int = 42,
    save_dir: str = "plots",
) -> dict:
    """Analyze probing accuracy grouped by early/mid/late layers."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    layers = sorted(hidden_states.keys())
    n = len(layers)

    # Split into thirds
    third = n // 3
    subsets = {
        "early": layers[:third],
        "mid": layers[third:2*third],
        "late": layers[2*third:],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = {}

    for subset_name, subset_layers in subsets.items():
        accs = []
        for layer in subset_layers:
            X = hidden_states[layer]
            scores = []
            for train_idx, test_idx in cv.split(X, y):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
                clf.fit(X_train, y[train_idx])
                scores.append(accuracy_score(y[test_idx], clf.predict(X_test)))
            accs.append(float(np.mean(scores)))
        results[subset_name] = {
            "layers": subset_layers,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "min_acc": float(np.min(accs)),
            "max_acc": float(np.max(accs)),
            "per_layer": dict(zip(subset_layers, accs)),
        }

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    means = [results[n]["mean_acc"] for n in names]
    stds = [results[n]["std_acc"] for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(names, means, yerr=stds, color=colors, edgecolor="black",
                  linewidth=0.5, capsize=5)
    ax.set_ylabel("Mean CV Accuracy", fontsize=12)
    ax.set_title(f"Layer Subset Analysis — {label_key}", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", fontsize=11)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"layer_subsets_{label_key}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Layer subset analysis → {path}")

    return results


# ---------------------------------------------------------------------------
# 4. Per-class analysis
# ---------------------------------------------------------------------------

def per_class_analysis(
    hidden_states: Dict[int, np.ndarray],
    labels: List[str],
    label_key: str,
    C: float = 1.0,
    cv_folds: int = 5,
    seed: int = 42,
    save_dir: str = "plots",
) -> dict:
    """Per-class precision/recall/F1 and confusion matrices across layers."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)
    layers = sorted(hidden_states.keys())

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Collect per-class metrics per layer
    per_layer_results = {}
    best_layer = None
    best_acc = -1

    for layer in layers:
        X = hidden_states[layer]
        all_y_true, all_y_pred = [], []

        for train_idx, test_idx in cv.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
            clf.fit(X_train, y[train_idx])
            all_y_true.extend(y[test_idx])
            all_y_pred.extend(clf.predict(X_test))

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        acc = float(accuracy_score(all_y_true, all_y_pred))
        if acc > best_acc:
            best_acc = acc
            best_layer = layer

        prec, rec, f1, support = precision_recall_fscore_support(
            all_y_true, all_y_pred, average=None, zero_division=0
        )
        cm = confusion_matrix(all_y_true, all_y_pred)

        per_layer_results[layer] = {
            "accuracy": acc,
            "per_class": {
                class_names[i]: {
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(len(class_names))
            },
            "confusion_matrix": cm.tolist(),
        }

    # Plot: per-class F1 over layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-class F1 over layers
    ax = axes[0]
    for i, cls in enumerate(class_names):
        f1s = [per_layer_results[l]["per_class"][cls]["f1"] for l in layers]
        ax.plot(layers, f1s, "o-", label=cls, linewidth=2, markersize=4)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"Per-Class F1 — {label_key}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Right: confusion matrix at best layer
    ax = axes[1]
    cm = np.array(per_layer_results[best_layer]["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (layer {best_layer})", fontsize=13)
    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"per_class_analysis_{label_key}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Per-class analysis → {path}")

    return {
        "class_names": class_names,
        "best_layer": best_layer,
        "per_layer": per_layer_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extended analysis for probing experiments")
    parser.add_argument("--hidden-states", type=str, required=True,
                        help="Path to hidden_states.npz (e.g. from --token-position last)")
    parser.add_argument("--hidden-states-alt", type=str, default=None,
                        help="Path to alternate hidden_states.npz (e.g. from --token-position mean) "
                             "for token position comparison")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to the dataset JSON file")
    parser.add_argument("--label-key", type=str, default="gender",
                        help="Which label key to analyze")
    parser.add_argument("--C-values", type=str, default="0.001,0.01,0.1,1.0,10.0,100.0",
                        help="Comma-separated C values for regularization sweep")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results/analysis")
    parser.add_argument("--plots-dir", type=str, default="plots/analysis")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Load data
    with open(args.data_file) as f:
        data = json.load(f)
    labels = data["labels"][args.label_key]
    print(f"Loaded {len(labels)} samples, classes: {dict(Counter(labels))}")

    # Load hidden states
    hs = load_hidden_states(args.hidden_states)
    print(f"Hidden states: {len(hs)} layers, {hs[list(hs.keys())[0]].shape}")

    C_values = [float(c) for c in args.C_values.split(",")]
    all_results = {}

    # --- 1. Token position comparison ---
    if args.hidden_states_alt:
        print("\n" + "=" * 60)
        print("  1. TOKEN POSITION COMPARISON")
        print("=" * 60)
        hs_alt = load_hidden_states(args.hidden_states_alt)
        res = compare_token_positions(
            hs, hs_alt, labels, args.label_key,
            cv_folds=args.cv_folds, seed=args.seed, save_dir=args.plots_dir,
        )
        all_results["token_position_comparison"] = res
    else:
        print("\n[Skip] Token position comparison (no --hidden-states-alt provided)")

    # --- 2. Regularization sweep ---
    print("\n" + "=" * 60)
    print("  2. REGULARIZATION SWEEP")
    print("=" * 60)
    res = regularization_sweep(
        hs, labels, args.label_key, C_values=C_values,
        cv_folds=args.cv_folds, seed=args.seed, save_dir=args.plots_dir,
    )
    all_results["regularization_sweep"] = res

    # --- 3. Layer subset analysis ---
    print("\n" + "=" * 60)
    print("  3. LAYER SUBSET ANALYSIS")
    print("=" * 60)
    res = layer_subset_analysis(
        hs, labels, args.label_key,
        cv_folds=args.cv_folds, seed=args.seed, save_dir=args.plots_dir,
    )
    all_results["layer_subsets"] = res
    for name, data_sub in res.items():
        print(f"  {name:6s}: acc={data_sub['mean_acc']:.3f} ± {data_sub['std_acc']:.3f} "
              f"(layers {data_sub['layers'][0]}-{data_sub['layers'][-1]})")

    # --- 4. Per-class analysis ---
    print("\n" + "=" * 60)
    print("  4. PER-CLASS ANALYSIS")
    print("=" * 60)
    res = per_class_analysis(
        hs, labels, args.label_key,
        cv_folds=args.cv_folds, seed=args.seed, save_dir=args.plots_dir,
    )
    all_results["per_class"] = {
        "class_names": res["class_names"],
        "best_layer": res["best_layer"],
        "best_layer_metrics": res["per_layer"][res["best_layer"]],
    }
    print(f"  Best layer: {res['best_layer']}")
    for cls, metrics in res["per_layer"][res["best_layer"]]["per_class"].items():
        print(f"    {cls}: prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
              f"f1={metrics['f1']:.3f} (n={metrics['support']})")

    # Save all results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = os.path.join(args.results_dir, f"analysis_{args.label_key}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n[Results] Saved to {results_path}")


if __name__ == "__main__":
    main()
