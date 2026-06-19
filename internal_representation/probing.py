"""
Step 3 & 4 — Linear Probing + Control-Task Evaluation.

Trains logistic regression classifiers on extracted hidden states to predict
expertise level. Also runs the selectivity control (randomized labels) to
verify that probe accuracy is not an artifact.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import ProbeConfig


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
    is_control: bool = False
    report: str = ""


@dataclass
class ProbeResults:
    """Container for all probe results across layers and attributes."""
    results: List[ProbeResult] = field(default_factory=list)

    def add(self, result: ProbeResult):
        self.results.append(result)

    def best_layer(self, attribute: str, classifier: str = "logistic", control: bool = False) -> Optional[ProbeResult]:
        """Return the result with the highest accuracy for a given attribute."""
        candidates = [
            r for r in self.results
            if r.attribute == attribute
            and r.classifier == classifier
            and r.is_control == control
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.accuracy)

    def to_dict(self) -> List[dict]:
        return [
            {
                "attribute": r.attribute,
                "layer": r.layer,
                "classifier": r.classifier,
                "accuracy": round(r.accuracy, 4),
                "f1_macro": round(r.f1_macro, 4),
                "cv_mean": round(r.cv_mean, 4),
                "cv_std": round(r.cv_std, 4),
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
) -> Tuple[float, float, float, float, str, object]:
    """
    Train a single linear probe and return metrics.

    Returns (accuracy, f1_macro, cv_mean, cv_std, classification_report_str, fitted_clf).
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

    from collections import Counter
    min_class_count = min(Counter(y_train).values())
    n_splits = max(2, min(config.cv_folds, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="accuracy")

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)

    return acc, f1, float(cv_scores.mean()), float(cv_scores.std()), str(report), clf


def train_probes(
    hidden_states: Dict[int, np.ndarray],
    labels: List[str],
    attribute_name: str,
    config: ProbeConfig,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[ProbeResults, Dict[int, object]]:
    """
    Train linear probes for a single attribute across all requested layers.

    Supports two evaluation modes (config.eval_mode):
      - "split": single stratified train/test split
      - "repeated_kfold": repeated stratified k-fold (more robust, default)
    """
    from sklearn.model_selection import train_test_split

    results = ProbeResults()
    classifiers: Dict[int, object] = {}

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    layers = sorted(hidden_states.keys())
    print(f"\n[Probing] Attribute: {attribute_name} | Classes: {list(le.classes_)} | Layers: {len(layers)}")
    print(f"  Eval mode: {config.eval_mode}")

    if config.eval_mode == "repeated_kfold":
        _train_probes_repeated_kfold(hidden_states, y_encoded, attribute_name, config, seed, layers, results, classifiers)
    else:
        _train_probes_split(hidden_states, y_encoded, attribute_name, config, test_size, seed, layers, results, classifiers)

    # Print summary for this attribute
    best_real = results.best_layer(attribute_name, "logistic", control=False)
    best_ctrl = results.best_layer(attribute_name, "logistic", control=True)
    if best_real and best_ctrl:
        print(f"  Best real probe:    layer {best_real.layer} — acc={best_real.accuracy:.3f}")
        print(f"  Best control probe: layer {best_ctrl.layer} — acc={best_ctrl.accuracy:.3f}")
        delta = best_real.accuracy - best_ctrl.accuracy
        print(f"  Selectivity gap:    {delta:+.3f}")

    return results, classifiers


def _train_probes_split(
    hidden_states: Dict[int, np.ndarray],
    y_encoded,
    attribute_name: str,
    config: ProbeConfig,
    test_size: float,
    seed: int,
    layers: List[int],
    results: ProbeResults,
    classifiers: Dict[int, object],
):
    """Single stratified train/test split evaluation."""
    from sklearn.model_selection import train_test_split

    for layer_idx in layers:
        X = hidden_states[layer_idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=seed,
        )

        # ── Real probe ────────────────────────────────────────────────
        acc, f1, cv_m, cv_s, report, clf = _train_and_eval_probe(
            X_train, y_train, X_test, y_test, config
        )
        results.add(ProbeResult(
            attribute=attribute_name, layer=layer_idx, classifier="logistic",
            accuracy=acc, f1_macro=f1, cv_mean=cv_m, cv_std=cv_s,
            is_control=False, report=report,
        ))
        classifiers[layer_idx] = clf

        # ── Control probe (randomized labels) ─────────────────────────
        rng = np.random.RandomState(seed)
        y_train_ctrl = rng.permutation(y_train)
        y_test_ctrl = rng.permutation(y_test)

        acc_c, f1_c, cv_m_c, cv_s_c, report_c, _ = _train_and_eval_probe(
            X_train, y_train_ctrl, X_test, y_test_ctrl, config
        )
        results.add(ProbeResult(
            attribute=attribute_name, layer=layer_idx, classifier="logistic",
            accuracy=acc_c, f1_macro=f1_c, cv_mean=cv_m_c, cv_std=cv_s_c,
            is_control=True, report=report_c,
        ))


def _train_probes_repeated_kfold(
    hidden_states: Dict[int, np.ndarray],
    y_encoded,
    attribute_name: str,
    config: ProbeConfig,
    seed: int,
    layers: List[int],
    results: ProbeResults,
    classifiers: Dict[int, object],
):
    """Repeated stratified k-fold evaluation for more robust estimates."""
    rkf = RepeatedStratifiedKFold(
        n_splits=config.cv_folds, n_repeats=config.n_repeats, random_state=seed
    )

    for layer_idx in layers:
        X = hidden_states[layer_idx]

        accs, f1s = [], []
        accs_ctrl, f1s_ctrl = [], []
        last_clf = None

        for train_idx, test_idx in rkf.split(X, y_encoded):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # ── Real probe ────────────────────────────────────────────
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(
                C=config.logistic_C, max_iter=config.max_iter,
                solver="lbfgs", random_state=42,
            )
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
            last_clf = clf

            # ── Control probe (shuffle within this fold) ──────────────
            rng = np.random.RandomState(seed + len(accs))
            y_train_ctrl = rng.permutation(y_train)
            y_test_ctrl = rng.permutation(y_test)

            clf_ctrl = LogisticRegression(
                C=config.logistic_C, max_iter=config.max_iter,
                solver="lbfgs", random_state=42,
            )
            clf_ctrl.fit(X_train_s, y_train_ctrl)
            y_pred_ctrl = clf_ctrl.predict(X_test_s)
            accs_ctrl.append(accuracy_score(y_test_ctrl, y_pred_ctrl))
            f1s_ctrl.append(f1_score(y_test_ctrl, y_pred_ctrl, average="macro", zero_division=0))

        # Aggregate across folds
        results.add(ProbeResult(
            attribute=attribute_name, layer=layer_idx, classifier="logistic",
            accuracy=round(float(np.mean(accs)), 4),
            f1_macro=round(float(np.mean(f1s)), 4),
            cv_mean=round(float(np.mean(accs)), 4),
            cv_std=round(float(np.std(accs)), 4),
            is_control=False,
        ))
        classifiers[layer_idx] = last_clf

        results.add(ProbeResult(
            attribute=attribute_name, layer=layer_idx, classifier="logistic",
            accuracy=round(float(np.mean(accs_ctrl)), 4),
            f1_macro=round(float(np.mean(f1s_ctrl)), 4),
            cv_mean=round(float(np.mean(accs_ctrl)), 4),
            cv_std=round(float(np.std(accs_ctrl)), 4),
            is_control=True,
        ))
