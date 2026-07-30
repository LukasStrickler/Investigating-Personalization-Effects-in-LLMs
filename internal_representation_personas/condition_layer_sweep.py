#!/usr/bin/env python3
"""Per-condition linear-probe accuracy sweep across all layers.

WHAT THIS PRODUCES
    The "Linear probe accuracy — all conditions" chart: one 5-fold-CV
    accuracy-vs-layer curve per condition, plus a shuffled-label control.

    Conditions (probe target = Gender in every case):
      * FULL         — original history text, unchanged.
      * NEUTRAL      — every indicator replaced by a gender-neutral word from
                       neutral_indicators.json (Movie/Hobby) or its angle-bracket
                       placeholder (Name/Artist, which have no neutral list).
      * ONLY_<TYPE>  — keep indicators of <TYPE> in place, replace every OTHER
                       indicator type with its placeholder. <TYPE> in
                       {Hobby, Movie, Name, Artist}.

    The question each ONLY_<TYPE> curve answers: how much Gender signal does a
    linear probe recover when ONLY that indicator type survives in the text?
    Movie/Hobby come from gender.csv; Name/Artist come from race.csv — so the
    Name/Artist curves also probe whether race-derived text leaks Gender.

RELATION TO THE ABLATION STAGE
    ``aggregate_word_analysis.run_ablation`` measures the shift in P(Male) when a
    single phrase is removed, using one fixed probe. This is different: it
    RETRAINS a Gender probe per layer (with 5-fold CV) on each condition's text,
    and plots accuracy-vs-layer. It reuses that module's ``load_targets`` /
    ``ablate_phrase`` for the text transforms and ``probing.train_probes`` for
    the CV probing (which already emits ``cv_mean`` per layer + a shuffled
    control), so the metric matches the chart's y-axis exactly.

    Only the FULL condition can reuse a prior run's ``hidden_states_personas.npz``
    (its text is unchanged). Every other condition alters the text, so its
    activations must be re-extracted from the model.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import joblib  # noqa: F401 — kept for parity with other stages / optional artifact dumps
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_word_analysis import ablate_phrase, load_targets
from config import ModelConfig, ProbeConfig
from dataset import _format_messages
from extraction import extract_hidden_states, load_hidden_states, load_model_and_tokenizer
from probing import train_probes

_HERE = Path(__file__).resolve().parent

# Indicator types that define the ONLY_<TYPE> conditions, and the display order
# / colours used in the chart. Movie+Hobby are gender indicators; Name+Artist
# are race indicators.
ONLY_TYPES = ["Movie", "Hobby", "Name", "Artist"]

_CONDITION_STYLE = {
    "FULL":                {"label": "FULL (all 4 indicators)",      "color": "#000000", "ls": "-",  "marker": "o"},
    "NEUTRAL":             {"label": "NEUTRAL (neutral words)",      "color": "#9467bd", "ls": "--", "marker": "v"},
    "NEUTRAL_GENERATED":   {"label": "NEUTRAL (regenerated convs)",  "color": "#e377c2", "ls": "--", "marker": "*"},
    "USER_ONLY":           {"label": "USER_ONLY (no asst turns)",    "color": "#2ca02c", "ls": "--", "marker": "P"},
    "NEUTRAL_USER_ONLY":   {"label": "NEUTRAL + no asst",           "color": "#8c564b", "ls": "--", "marker": "X"},
    "TEMPLATE_ONLY":       {"label": "TEMPLATE_ONLY (chance anchor)","color": "#aaaaaa", "ls": ":",  "marker": ""},
    "ONLY_MOVIE":          {"label": "ONLY_MOVIE",                   "color": "#d62728", "ls": "--", "marker": "s"},
    "ONLY_HOBBY":          {"label": "ONLY_HOBBY",                   "color": "#4C72B0", "ls": "--", "marker": ">"},
    "ONLY_NAME":           {"label": "ONLY_NAME",                    "color": "#17a2a2", "ls": "--", "marker": "D"},
    "ONLY_ARTIST":         {"label": "ONLY_ARTIST",                  "color": "#e6a817", "ls": "--", "marker": "+"},
    "ONLY_MOVIE_USER_ONLY":{"label": "ONLY_MOVIE + no asst",        "color": "#d62728", "ls": ":",  "marker": "s"},
    "ONLY_HOBBY_USER_ONLY":{"label": "ONLY_HOBBY + no asst",        "color": "#4C72B0", "ls": ":",  "marker": ">"},
}


def _placeholder(target: dict) -> str:
    """The neutral placeholder an indicator is replaced with (e.g. '<Hobby>')."""
    return f"<{target['indicator_name']}>"


def _load_neutral_words(neutral_path: Path | None = None) -> dict[str, list[str]]:
    """Load neutral replacement words from neutral_indicators.json.

    Returns a dict with lowercase indicator_name keys, e.g.
    {"movie": [...], "hobby": [...]}.
    """
    path = neutral_path or (_HERE / "neutral_indicators.json")
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    # Keys in the file are plural ("movies", "hobbies"); normalise to singular
    # to match indicator_name values ("Movie" -> "movie").
    # Map plural file keys to the singular indicator_name used in targets.
    # Hardcode the two known keys from neutral_indicators.json.
    _PLURAL_TO_SINGULAR = {"movies": "movie", "hobbies": "hobby"}
    mapping: dict[str, list[str]] = {}
    for key, words in raw.items():
        if isinstance(words, list):
            singular = _PLURAL_TO_SINGULAR.get(key.lower(), key.lower().rstrip("s"))
            mapping[singular] = words
    return mapping


def _neutral_replacement(target: dict, neutral_words: dict[str, list[str]], index: int) -> str:
    """Return a neutral replacement word for an indicator in the NEUTRAL condition.

    For indicator types that have neutral counterparts (Movie, Hobby), cycle
    through the neutral word list deterministically based on history index so
    that different histories get different neutral words. For other types
    (Name, Artist) fall back to the angle-bracket placeholder.
    """
    ind_name = target["indicator_name"].lower()
    words = neutral_words.get(ind_name)
    if words:
        return words[index % len(words)]
    return _placeholder(target)


def build_condition_histories(
    histories: list[list[dict]],
    targets: list[dict],
    condition: str,
    neutral_path: Path | None = None,
) -> list[str]:
    """Return the formatted (string) histories for one condition.

    * FULL              — untouched.
    * USER_ONLY         — strip all assistant turns; only user turns remain.
                          Eliminates artist-pronoun and greeting-style leaks.
    * NEUTRAL           — replace every indicator with a gender-neutral word
                          (from neutral_indicators.json for Movie/Hobby; angle-bracket
                          placeholder for Name/Artist which have no neutral list).
    * NEUTRAL_USER_ONLY — NEUTRAL replacements + assistant turns stripped.
                          The cleanest possible condition: no indicator words,
                          no assistant-side cultural/gender leakage.
    * TEMPLATE_ONLY     — all indicator values replaced with placeholders AND
                          assistant turns stripped. Both genders produce identical
                          text — guaranteed chance-level negative control.
    * ONLY_<TYPE>       — keep indicators of <TYPE> verbatim; replace every OTHER
                          indicator type with its angle-bracket placeholder.
                          Replacement covers ALL turns including assistant replies.
    * ONLY_MOVIE_USER_ONLY / ONLY_HOBBY_USER_ONLY
                        — single gender-direct indicator kept, assistant turns
                          stripped. Isolates movie/hobby signal without any
                          assistant-side artist/greeting leakage.

    Controls are never touched. Replacement uses longest-phrase-first order so
    'romantic drama' is handled before 'drama'.
    """
    indicators = [t for t in targets if t["target_type"].endswith("indicator")]
    if condition == "FULL":
        return [_format_messages(messages) for messages in histories]

    if condition == "USER_ONLY":
        return [
            _format_messages([m for m in messages if m.get("role") != "assistant"])
            for messages in histories
        ]

    if condition == "TEMPLATE_ONLY":
        # All indicators → placeholder, no assistant turns.  Both genders produce
        # identical text — serves as the guaranteed chance-level anchor.
        ordered = sorted(indicators, key=lambda t: len(t["phrase"]), reverse=True)
        partial_passes = _build_partial_passes(ordered)
        out: list[str] = []
        for messages in histories:
            user_only = [m for m in messages if m.get("role") != "assistant"]
            current = user_only
            for target in ordered:
                current = ablate_phrase(current, target["phrase"], _placeholder(target))
            for token, placeholder in partial_passes:
                current = ablate_phrase(current, token, placeholder)
            out.append(_format_messages(current))
        return out

    if condition == "NEUTRAL_USER_ONLY":
        neutral_words = _load_neutral_words(neutral_path)
        ordered = sorted(indicators, key=lambda t: len(t["phrase"]), reverse=True)
        partial_passes = _build_partial_passes(ordered)
        out = []
        for hist_idx, messages in enumerate(histories):
            user_only = [m for m in messages if m.get("role") != "assistant"]
            current = user_only
            for target in ordered:
                current = ablate_phrase(
                    current, target["phrase"],
                    _neutral_replacement(target, neutral_words, hist_idx),
                )
            for token, placeholder in partial_passes:
                current = ablate_phrase(current, token, placeholder)
            out.append(_format_messages(current))
        return out

    if condition in ("ONLY_MOVIE_USER_ONLY", "ONLY_HOBBY_USER_ONLY"):
        keep_type = condition[len("ONLY_"):].split("_USER_ONLY")[0].title()
        to_replace = [t for t in indicators if t["indicator_name"].title() != keep_type]
        ordered = sorted(to_replace, key=lambda t: len(t["phrase"]), reverse=True)
        partial_passes = _build_partial_passes(ordered)
        out = []
        for messages in histories:
            user_only = [m for m in messages if m.get("role") != "assistant"]
            current = user_only
            for target in ordered:
                current = ablate_phrase(current, target["phrase"], _placeholder(target))
            for token, placeholder in partial_passes:
                current = ablate_phrase(current, token, placeholder)
            out.append(_format_messages(current))
        return out

    if condition == "NEUTRAL":
        to_replace = indicators
        neutral_words = _load_neutral_words(neutral_path)
    elif condition.startswith("ONLY_"):
        keep_type = condition[len("ONLY_"):].title()
        to_replace = [t for t in indicators if t["indicator_name"].title() != keep_type]
        neutral_words = {}
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Longest phrase first to avoid partial-substring clobbering.
    ordered = sorted(to_replace, key=lambda t: len(t["phrase"]), reverse=True)
    partial_passes = _build_partial_passes(ordered)

    out: list[str] = []
    for hist_idx, messages in enumerate(histories):
        current = messages
        for target in ordered:
            if condition == "NEUTRAL":
                replacement = _neutral_replacement(target, neutral_words, hist_idx)
            else:
                replacement = _placeholder(target)
            current = ablate_phrase(current, target["phrase"], replacement)
        for token, placeholder in partial_passes:
            current = ablate_phrase(current, token, placeholder)
        out.append(_format_messages(current))
    return out


def _build_partial_passes(ordered: list[dict]) -> list[tuple[str, str]]:
    """Build (token, placeholder) pairs for partial echoes of multi-word Name/Artist indicators.

    The model often echoes only a first name (e.g. "Hello Eva!" when the phrase
    was "Eva Svobodová"). This second pass catches those residual tokens.
    Restricted to Name/Artist (not Movie/Hobby) since those are the ones echoed.
    Skips single-character tokens only (e.g. bare punctuation).
    """
    seen: set[str] = set()
    passes: list[tuple[str, str]] = []
    for target in ordered:
        if target["indicator_name"] not in ("Name", "Artist"):
            continue
        words = target["phrase"].split()
        if len(words) < 2:
            continue
        placeholder = _placeholder(target)
        for word in sorted(words, key=len, reverse=True):
            if len(word) < 2:
                continue
            wl = word.casefold()
            if wl not in seen:
                seen.add(wl)
                passes.append((word, placeholder))
    return passes


def _cv_curve(results, control: bool) -> tuple[list[int], list[float], list[float], list[float]]:
    """Extract (layers, cv_mean, cv_std, mean_margin) for the real or control probe."""
    rows = [
        r for r in results.results
        if r.classifier == "logistic" and r.is_control == control
    ]
    rows.sort(key=lambda r: r.layer)
    return (
        [r.layer for r in rows],
        [r.cv_mean for r in rows],
        [r.cv_std for r in rows],
        [r.mean_margin for r in rows],
    )


def run_condition_sweep(
    dataset_path: str | Path,
    out_dir: str | Path,
    model_name: str,
    device_map: str,
    dtype: str,
    mapping_paths: list[str | Path],
    control_words: list[str],
    full_hidden_path: str | Path | None = None,
    conditions: list[str] | None = None,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
    checkpoint_cb=None,
    neutral_path: str | Path | None = None,
    neutral_dataset_path: str | Path | None = None,
) -> dict:
    """Run the per-condition, per-layer 5-fold-CV Gender-probe sweep.

    ``full_hidden_path`` — if given and it exists, the FULL condition reuses
    those cached hidden states instead of re-extracting (its text is unchanged).
    All other conditions are always re-extracted.

    Resumability: each condition's curve is checkpointed to
    ``checkpoint_dir`` (a JSON per condition) as soon as it completes. On a
    restart (e.g. Modal spot preemption), already-checkpointed conditions are
    loaded and skipped, so at most one condition's compute is ever repeated.
    ``checkpoint_cb(condition)`` is called after each checkpoint is written
    (e.g. to commit the volume). Defaults ``checkpoint_dir`` to ``out_dir``.

    Writes ``condition_layer_accuracy.csv`` and ``condition_layer_accuracy.png``
    to ``out_dir`` and returns a metadata dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if conditions is None:
        conditions = (
            ["FULL", "USER_ONLY", "NEUTRAL", "NEUTRAL_USER_ONLY", "TEMPLATE_ONLY"]
            + [f"ONLY_{t.upper()}" for t in ONLY_TYPES]
            + ["ONLY_MOVIE_USER_ONLY", "ONLY_HOBBY_USER_ONLY"]
        )

    with open(dataset_path, encoding="utf-8") as handle:
        dataset = json.load(handle)
    if "Gender" not in dataset["labels"]:
        raise ValueError(f"Sweep requires 'Gender' labels; dataset has: {sorted(dataset['labels'])}")

    labels_all = dataset["labels"]["Gender"]
    keep = [i for i, lbl in enumerate(labels_all) if lbl is not None]
    histories = [dataset["conversations_chat"][i] for i in keep]
    labels = [labels_all[i] for i in keep]
    print(f"[sweep] {len(histories)} labelled histories | {dict(Counter(labels))}")

    targets = load_targets([Path(p) for p in mapping_paths], control_words)

    model_config = ModelConfig(
        model_name=str(model_name), device_map=device_map,
        torch_dtype=dtype, max_seq_length=max_seq_length,
    )
    probe_config = ProbeConfig(layers=None, token_position="last")  # all layers

    model = tokenizer = None  # lazy: only load if a condition needs extraction

    curves: dict[str, dict] = {}
    for condition in conditions:
        ckpt_path = ckpt_dir / f"{condition}.json"
        if ckpt_path.exists():
            # Resume: a prior (possibly preempted) run already finished this one.
            curves[condition] = json.loads(ckpt_path.read_text())
            print(f"[sweep] === condition: {condition} === (resumed from checkpoint)")
            continue

        print(f"\n[sweep] === condition: {condition} ===")
        # Each condition trains on its own labels. Only NEUTRAL_GENERATED uses a
        # different dataset (freshly regenerated neutral conversations); every
        # other condition shares the primary dataset's histories/labels.
        cond_labels = labels
        reuse_full = (
            condition == "FULL"
            and full_hidden_path is not None
            and Path(full_hidden_path).exists()
        )
        if reuse_full:
            print(f"[sweep] FULL reuses cached hidden states: {full_hidden_path}")
            hidden = load_hidden_states(str(full_hidden_path))
            # The cached npz has one row per dataset history; align to the
            # (non-null-label) subset used here.
            hidden = {layer: values[keep] for layer, values in hidden.items()}
        elif condition == "NEUTRAL_GENERATED":
            if not neutral_dataset_path or not Path(neutral_dataset_path).exists():
                raise FileNotFoundError(
                    f"NEUTRAL_GENERATED needs neutral_dataset_path; got {neutral_dataset_path!r}"
                )
            with open(neutral_dataset_path, encoding="utf-8") as handle:
                ndataset = json.load(handle)
            nlabels_all = ndataset["labels"]["Gender"]
            nkeep = [i for i, lbl in enumerate(nlabels_all) if lbl is not None]
            nhistories = [ndataset["conversations_chat"][i] for i in nkeep]
            cond_labels = [nlabels_all[i] for i in nkeep]
            print(f"[sweep] NEUTRAL_GENERATED: {len(nhistories)} regenerated histories | {dict(Counter(cond_labels))}")
            texts = [_format_messages(messages) for messages in nhistories]
            if model is None:
                model, tokenizer = load_model_and_tokenizer(model_config)
            hidden = extract_hidden_states(
                model, tokenizer, texts, model_config, probe_config, batch_size=batch_size
            )
        else:
            texts = build_condition_histories(
                histories, targets, condition,
                neutral_path=Path(neutral_path) if neutral_path else None,
            )
            if model is None:
                model, tokenizer = load_model_and_tokenizer(model_config)
            hidden = extract_hidden_states(
                model, tokenizer, texts, model_config, probe_config, batch_size=batch_size
            )

        results, _ = train_probes(
            hidden, cond_labels, "Gender", probe_config,
            test_size=0.2, seed=seed,
        )
        r_layers, r_mean, r_std, r_margin = _cv_curve(results, control=False)
        c_layers, c_mean, c_std, c_margin = _cv_curve(results, control=True)
        curve = {
            "layers": r_layers, "cv_mean": r_mean, "cv_std": r_std,
            "mean_margin": r_margin,
            "control_cv_mean": c_mean, "control_cv_std": c_std,
            "control_mean_margin": c_margin,
        }
        curves[condition] = curve
        # Checkpoint IMMEDIATELY so a preemption after this point does not
        # re-run this condition on restart.
        ckpt_path.write_text(json.dumps(curve))
        print(f"[sweep] checkpointed condition '{condition}' -> {ckpt_path}")
        if checkpoint_cb is not None:
            checkpoint_cb(condition)

    _write_csv(out_dir / "condition_layer_accuracy.csv", curves)
    _plot(out_dir / "condition_layer_accuracy.png", curves, conditions)

    metadata = {
        "histories": len(histories),
        "label_counts": Counter(labels),
        "conditions": conditions,
        "reused_full_hidden": bool(
            full_hidden_path and Path(full_hidden_path).exists()
        ),
        "metric": "5-fold stratified CV accuracy (cv_mean) per layer",
    }
    with (out_dir / "condition_layer_accuracy_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=dict)
    print("\n" + json.dumps(metadata, indent=2, default=dict))
    return metadata


def _write_csv(path: Path, curves: dict) -> None:
    rows = []
    for condition, data in curves.items():
        for i, (layer, m, s, mg, cm, cs, cmg) in enumerate(zip(
            data["layers"], data["cv_mean"], data["cv_std"],
            data.get("mean_margin", [0.0] * len(data["layers"])),
            data["control_cv_mean"], data["control_cv_std"],
            data.get("control_mean_margin", [0.0] * len(data["layers"])),
        )):
            rows.append({
                "condition": condition, "layer": layer,
                "cv_mean": m, "cv_std": s, "mean_margin": mg,
                "control_cv_mean": cm, "control_cv_std": cs,
                "control_mean_margin": cmg,
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, curves: dict, conditions: list[str]) -> None:
    fig, (ax_acc, ax_margin) = plt.subplots(2, 1, figsize=(8, 11), sharex=True)

    ctrl_src = "FULL" if "FULL" in curves else conditions[0]

    for condition in conditions:
        if condition not in curves:
            continue
        style = _CONDITION_STYLE.get(condition, {"label": condition, "color": None, "ls": "--", "marker": "."})
        data = curves[condition]
        kwargs = dict(label=style["label"], color=style["color"],
                      linestyle=style["ls"], marker=style["marker"],
                      markersize=4, linewidth=1.2)
        ax_acc.plot(data["layers"], data["cv_mean"], **kwargs)
        if "mean_margin" in data:
            ax_margin.plot(data["layers"], data["mean_margin"], **kwargs)

    # Shuffled-label control curves
    ctrl = curves[ctrl_src]
    ctrl_kwargs = dict(label="Shuffled-label control", color="#888888",
                       linestyle=":", linewidth=1.0)
    ax_acc.plot(ctrl["layers"], ctrl["control_cv_mean"], **ctrl_kwargs)
    if "control_mean_margin" in ctrl:
        ax_margin.plot(ctrl["layers"], ctrl["control_mean_margin"], **ctrl_kwargs)

    ax_acc.axhline(0.5, color="#cccccc", linewidth=0.8, linestyle=":")
    ax_acc.set_ylabel("5-fold CV accuracy")
    ax_acc.set_title("Linear probe accuracy — all conditions\n(fresh probe retrained per condition; probe target = Gender)")
    ax_acc.legend(loc="center right", fontsize=9)

    ax_margin.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle=":")
    ax_margin.set_xlabel("Layer")
    ax_margin.set_ylabel("Mean confidence margin\n(top prob − 2nd prob, test set)")
    ax_margin.set_title("Probe confidence margin — sensitive even when accuracy = 1.0")
    ax_margin.legend(loc="center right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
