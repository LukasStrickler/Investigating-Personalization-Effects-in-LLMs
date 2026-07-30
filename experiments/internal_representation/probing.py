"""
Step 3 & 4 — Linear Probing + Control-Task Evaluation.

Trains logistic regression classifiers on extracted hidden states to predict
persona attributes. Also runs the selectivity control (randomized labels) to
verify that probe accuracy is not an artifact.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from config import ProbeConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Data class for probe results
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Holds evaluation metrics for a single probe."""

    attribute: str
    layer: int
    classifier: str  # "logistic" or "svm"
    accuracy: float
    f1_macro: float
    cv_mean: float
    cv_std: float
    mean_margin: float = 0.0  # mean(max_prob - second_max_prob) on test set
    is_control: bool = False
    report: str = ""


@dataclass
class ProbeArtifact:
    """Fitted probe plus the exact held-out indices used for evaluation."""

    scaler: StandardScaler
    classifier: LogisticRegression
    label_encoder: LabelEncoder
    train_indices: np.ndarray
    test_indices: np.ndarray
    layer: int
    source_indices: np.ndarray | None = None


@dataclass
class ProbeResults:
    """Container for all probe results across layers and attributes."""

    results: list[ProbeResult] = field(default_factory=list)

    def add(self, result: ProbeResult):
        self.results.append(result)

    def best_layer(
        self, attribute: str, classifier: str = "logistic", control: bool = False
    ) -> ProbeResult | None:
        """Return the result with the highest accuracy for a given attribute."""
        candidates = [
            r
            for r in self.results
            if r.attribute == attribute and r.classifier == classifier and r.is_control == control
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.accuracy)

    def to_dict(self) -> list[dict]:
        return [
            {
                "attribute": r.attribute,
                "layer": r.layer,
                "classifier": r.classifier,
                "accuracy": round(r.accuracy, 4),
                "f1_macro": round(r.f1_macro, 4),
                "cv_mean": round(r.cv_mean, 4),
                "cv_std": round(r.cv_std, 4),
                "mean_margin": round(r.mean_margin, 4),
                "is_control": r.is_control,
            }
            for r in self.results
        ]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Probing] Saved results to {path}")


# ---------------------------------------------------------------------------
# Core probing logic
# ---------------------------------------------------------------------------


def _train_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ProbeConfig,
) -> tuple[float, float, float, float, float, str, StandardScaler, LogisticRegression]:
    """
    Train a single linear probe and return metrics.

    Returns (accuracy, f1_macro, cv_mean, cv_std, mean_margin,
             classification_report_str, scaler, fitted_clf).
    mean_margin = mean(top_prob - second_prob) over test samples — stays
    informative even when accuracy saturates at 1.0.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=config.logistic_C,
        max_iter=config.max_iter,
        solver="lbfgs",
        random_state=42,
    )

    class_counts = np.bincount(y_train)
    n_splits = min(config.cv_folds, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Too few training samples per class; increase --samples to at least 3")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="accuracy")

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)  # (n_test, n_classes)
    sorted_proba = np.sort(proba, axis=1)
    mean_margin = float(np.mean(sorted_proba[:, -1] - sorted_proba[:, -2]))

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)

    return (
        acc,
        f1,
        float(cv_scores.mean()),
        float(cv_scores.std()),
        mean_margin,
        str(report),
        scaler,
        clf,
    )


def train_probes(
    hidden_states: dict[int, np.ndarray],
    labels: list[str],
    attribute_name: str,
    config: ProbeConfig,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[ProbeResults, dict[int, ProbeArtifact]]:
    """
    Train linear probes for a single attribute across all requested layers.

    Parameters
    ----------
    hidden_states : dict layer_idx -> (n_samples, hidden_dim) array
    labels : list of string labels (length = n_samples)
    attribute_name : name of the attribute being probed
    config : probe configuration
    test_size : fraction of data to hold out for evaluation
    seed : random seed for reproducibility

    Returns
    -------
    results : ProbeResults containing metrics for every layer
    classifiers : dict mapping layer_idx -> fitted classifier (for steering)
    """
    from sklearn.model_selection import train_test_split

    results = ProbeResults()
    artifacts: dict[int, ProbeArtifact] = {}

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    layers = sorted(hidden_states.keys())
    print(
        f"\n[Probing] Attribute: {attribute_name} | Classes: {list(le.classes_)} | Layers: {len(layers)}"
    )

    for layer_idx in layers:
        X = hidden_states[layer_idx]
        all_indices = np.arange(len(y_encoded))
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            stratify=y_encoded,
            random_state=seed,
        )
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

        # ── Real probe ────────────────────────────────────────────────
        acc, f1, cv_m, cv_s, margin, report, scaler, clf = _train_and_eval_probe(
            X_train, y_train, X_test, y_test, config
        )
        results.add(
            ProbeResult(
                attribute=attribute_name,
                layer=layer_idx,
                classifier="logistic",
                accuracy=acc,
                f1_macro=f1,
                cv_mean=cv_m,
                cv_std=cv_s,
                mean_margin=margin,
                is_control=False,
                report=report,
            )
        )
        artifacts[layer_idx] = ProbeArtifact(
            scaler=scaler,
            classifier=clf,
            label_encoder=le,
            train_indices=train_indices,
            test_indices=test_indices,
            layer=layer_idx,
        )

        # ── Control probe (randomized labels) ─────────────────────────
        rng = np.random.RandomState(seed)
        y_shuffled = rng.permutation(y_encoded)
        y_train_ctrl = y_shuffled[train_indices]
        y_test_ctrl = y_shuffled[test_indices]

        acc_c, f1_c, cv_m_c, cv_s_c, margin_c, report_c, _, _ = _train_and_eval_probe(
            X_train, y_train_ctrl, X_test, y_test_ctrl, config
        )
        results.add(
            ProbeResult(
                attribute=attribute_name,
                layer=layer_idx,
                classifier="logistic",
                accuracy=acc_c,
                f1_macro=f1_c,
                cv_mean=cv_m_c,
                cv_std=cv_s_c,
                mean_margin=margin_c,
                is_control=True,
                report=report_c,
            )
        )

    # Print summary for this attribute
    best_real = results.best_layer(attribute_name, "logistic", control=False)
    best_ctrl = results.best_layer(attribute_name, "logistic", control=True)
    if best_real and best_ctrl:
        print(f"  Best real probe:    layer {best_real.layer} — acc={best_real.accuracy:.3f}")
        print(f"  Best control probe: layer {best_ctrl.layer} — acc={best_ctrl.accuracy:.3f}")
        delta = best_real.accuracy - best_ctrl.accuracy
        print(f"  Selectivity gap:    {delta:+.3f}")

    return results, artifacts
