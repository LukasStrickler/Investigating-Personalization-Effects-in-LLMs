#!/usr/bin/env python3
"""Ablate every configured Gender indicator phrase plus neutral controls."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ModelConfig, ProbeConfig
from dataset import _format_messages
from extraction import extract_hidden_states, load_model_and_tokenizer


# These words come from the fixed Gender question template, not from <Movie>
# or <Hobby>. They occur in both classes and therefore form neutral controls.
DEFAULT_CONTROL_WORDS = [
    "really", "enjoy", "watching", "movies", "outside",
    "spend", "time", "someone", "background", "structure",
]


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)


def user_text(messages: list[dict]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )


def ablate_phrase(messages: list[dict], phrase: str, replacement: str = "") -> list[dict]:
    """Replace or remove a phrase across ALL turns (user and assistant).

    For indicators, pass replacement=<IndicatorName> to restore the neutral
    template placeholder (e.g. 'action' -> '<Movie>'). For control words that
    have no placeholder, pass replacement="" to delete them.

    Assistant turns often echo the indicator verbatim (e.g. the user says
    "I am Pavel" and the reply opens "Hello Pavel!"), so scrubbing only the
    user turn would leak the indicator into the probe's context. We therefore
    replace every occurrence in every turn. This does NOT catch forms the model
    *invents* (inflections, translations, pronouns) — only verbatim occurrences
    of the configured phrase — but that is the defensible, string-exact bound.
    """
    result = deepcopy(messages)
    pattern = phrase_pattern(phrase)
    for message in result:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if replacement:
            message["content"] = pattern.sub(replacement, content)
        else:
            message["content"] = re.sub(r"\s+([,.!?])", r"\1", pattern.sub("", content))
    return result


def _dimension_from_path(mapping_path: Path) -> str:
    """Infer the dimension tag ('Gender', 'Region', ...) from the CSV filename."""
    return mapping_path.stem.replace("_", " ").title()


def load_targets(mapping_paths: list[Path], controls: list[str]) -> list[dict[str, str]]:
    """Load indicator phrases from one or more dimension-value mapping CSVs.

    Handles the two header schemas in the repo: ``gender.csv`` uses
    ``Dimension_value`` (underscore) while ``region.csv`` uses ``Dimension value``
    (space, BOM-prefixed — stripped by ``utf-8-sig``). Gender rows keep their
    Male/Female direction in ``configured_gender``; rows from any other
    dimension get ``configured_gender="<Dimension>"`` (e.g. ``"Region"``) so the
    effect plots render them as their own group rather than mis-slotting them
    as Male/Female. Phrases seen in an earlier CSV are not added twice.
    """
    targets: list[dict[str, str]] = []
    seen_phrases: set[str] = set()
    for mapping_path in mapping_paths:
        dimension = _dimension_from_path(mapping_path)
        with mapping_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                # Normalize the dimension-value column across schemas.
                dimension_value = (row.get("Dimension_value") or row.get("Dimension value") or "").strip()
                phrase = row["Indicator_value"].strip()
                if phrase.casefold() in seen_phrases:
                    continue  # de-dup across CSVs; keep the first occurrence
                seen_phrases.add(phrase.casefold())
                # Gender keeps its Male/Female direction; other dimensions
                # (Region) have no gender direction, so group them by dimension.
                configured_gender = dimension_value if dimension == "Gender" else dimension
                targets.append({
                    "phrase": phrase,
                    "target_type": f"{dimension} indicator",
                    "dimension": dimension,
                    "dimension_value": dimension_value,
                    "configured_gender": configured_gender,
                    "indicator_name": row["Indicator_name"].strip(),
                })
    for word in controls:
        if word.casefold() in seen_phrases:
            raise ValueError(f"Control word is also an indicator: {word}")
        targets.append({
            "phrase": word,
            "target_type": "Neutral template control",
            "dimension": "Control",
            "dimension_value": "Control",
            "configured_gender": "Control",
            "indicator_name": "Fixed template word",
        })
    return targets


def matched_indicators(text: str, indicators: list[dict[str, str]]) -> set[str]:
    """Return longest configured matches so `drama` does not match `romantic drama`."""
    occupied: list[tuple[int, int]] = []
    matched: set[str] = set()
    for target in sorted(indicators, key=lambda row: len(row["phrase"]), reverse=True):
        for match in phrase_pattern(target["phrase"]).finditer(text):
            span = match.span()
            if not any(span[0] < end and start < span[1] for start, end in occupied):
                occupied.append(span)
                matched.add(target["phrase"])
    return matched


def save_presence_table(path: Path, targets: list[dict], row_targets: list[set[str]], labels: list[str]) -> list[dict]:
    totals = Counter(labels)
    rows = []
    for target in targets:
        indices = [index for index, values in enumerate(row_targets) if target["phrase"] in values]
        counts = Counter(labels[index] for index in indices)
        count = len(indices)
        rows.append({
            **target,
            "histories_containing": count,
            "female_count": counts["Female"],
            "male_count": counts["Male"],
            "p_female_given_present": counts["Female"] / count if count else "",
            "p_male_given_present": counts["Male"] / count if count else "",
            "female_document_frequency": counts["Female"] / totals["Female"],
            "male_document_frequency": counts["Male"] / totals["Male"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            raise ValueError(f"No presence rows to write to {path} — the selected histories matched no targets.")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_presence(rows: list[dict], path: Path) -> None:
    # Only plot targets that actually occur in the selected histories — absent
    # targets have an empty ("") conditional-probability field.
    present = [row for row in rows if row["histories_containing"]]
    if not present:
        return
    ordered = sorted(present, key=lambda row: (row["dimension"] == "Control", row["dimension"], row["configured_gender"], row["indicator_name"], row["phrase"]))
    y = np.arange(len(ordered))
    female = [float(row["p_female_given_present"]) for row in ordered]
    male = [float(row["p_male_given_present"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.32 * len(ordered))))
    ax.barh(y, male, label="Male", color="#d97706")
    ax.barh(y, female, left=male, label="Female", color="#2878b5")
    ax.set_yticks(y, [row["phrase"] for row in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(true Gender label | phrase is present)")
    ax.set_title(f"Gender-label prevalence for {len(ordered)} present indicators/controls")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ablation_scatter(detail_rows: list[dict], path: Path, artifact_layer: int) -> None:
    """Scatter of baseline vs ablated P(Male) per (history, phrase) pair.

    Points below the diagonal mean the indicator raised P(Male); replacing it
    with the neutral placeholder dropped the probe's confidence in Male.
    Points above the diagonal mean the indicator suppressed P(Male).
    """
    color_map = {"Male": "#d97706", "Female": "#2878b5", "Region": "#59a14f", "Control": "#999999"}
    label_map = {"Male": "Male indicator → ↑P(Male) when present",
                 "Female": "Female indicator → ↓P(Male) when present",
                 "Region": "Region/region indicator → expect ≈ no change",
                 "Control": "Neutral control"}

    fig, ax = plt.subplots(figsize=(7, 7))
    for gender_type, color in color_map.items():
        rows = [r for r in detail_rows if r.get("configured_gender") == gender_type]
        if rows:
            ax.scatter(
                [r["baseline_p_male"] for r in rows],
                [r["ablated_p_male"] for r in rows],
                c=color, alpha=0.35, s=16, label=label_map[gender_type], linewidths=0,
            )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="no change (y = x)")
    ax.set_xlabel("Baseline P(Male)  [original text with indicator]")
    ax.set_ylabel("P(Male) after replacing indicator with placeholder")
    ax.set_title(
        f"Does replacing the indicator change the probe? (layer {artifact_layer})\n"
        "Below diagonal: yes — indicator presence raised P(Male)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_effect_by_gender_group(detail_rows: list[dict], path: Path, artifact_layer: int) -> None:
    """Distribution of ablation effects grouped by Male / Female / Control.

    Each data point is the effect for one (history, phrase) pair.
    Positive effect = indicator raised P(Male); negative = it suppressed it.
    If the probe reads the indicators, Male indicators should cluster positive
    and Female indicators should cluster negative.
    """
    groups: dict[str, list[float]] = {"Male": [], "Female": [], "Region": [], "Control": []}
    for row in detail_rows:
        key = row.get("configured_gender", "Control")
        if key in groups:
            groups[key].append(row["effect_of_phrase_presence_on_p_male"])

    order = [g for g in ("Male", "Female", "Region", "Control") if groups[g]]
    data = [groups[g] for g in order]
    color_map = {"Male": "#d97706", "Female": "#2878b5", "Region": "#59a14f", "Control": "#999999"}
    xlabels = {
        "Male": "Male indicators\n(expect: positive)",
        "Female": "Female indicators\n(expect: negative)",
        "Region": "Region/region indicators\n(expect: ≈ 0 if probe is\ngender-pure)",
        "Control": "Neutral controls\n(expect: ≈ 0)",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, positions=range(1, len(order) + 1), showmedians=True, showextrema=True)
    for body, group in zip(parts["bodies"], order):
        body.set_facecolor(color_map[group])
        body.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(1, len(order) + 1), [xlabels[g] for g in order])
    ax.set_ylabel("Effect on probe P(Male)  [baseline − ablated]\n(positive = indicator presence raised P(Male))")
    ax.set_title(
        f"Does the probe read gender indicators? (layer {artifact_layer})\n"
        "Male indicators should shift P(Male) up; Female indicators should shift it down"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _balanced_keep_indices(labels: list, subset_per_class: int, seed: int) -> list[int]:
    """Balanced (equal Female/Male) ascending index list into the dataset.

    Drops null-label rows, downsamples the larger class to match the smaller,
    then applies the optional per-class cap. Deterministic given ``seed`` so
    reruns pick the same histories. Returns indices in ascending order so
    dataset lists, npz rows, and baseline probabilities stay aligned.
    """
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        if label is not None:
            by_label[label].append(index)
    if set(by_label) < {"Female", "Male"}:
        raise ValueError(f"Ablation needs both Female and Male histories; got {dict(Counter(labels))}")

    per_class = min(len(by_label["Female"]), len(by_label["Male"]))
    if subset_per_class > 0:
        per_class = min(per_class, subset_per_class)

    rng = random.Random(seed)
    keep: list[int] = []
    for label in ("Female", "Male"):
        indices = by_label[label]
        keep.extend(sorted(rng.sample(indices, per_class)) if len(indices) > per_class else indices)
    return sorted(keep)


def run_ablation(
    dataset_path: str | Path,
    run_dir: str | Path,
    plot_dir: str | Path,
    model_name: str,
    device_map: str,
    dtype: str,
    mapping_paths: list[str | Path],
    control_words: list[str],
    subset_per_class: int = 0,
    batch_size: int = 4,
    seed: int = 42,
) -> dict:
    """Ablate gender + region indicator phrases against the Gender probe.

    Loads the probe, hidden states and normalized dataset from ``run_dir`` /
    ``dataset_path``, balances the histories to equal Female/Male counts
    (downsampling the larger class; optionally capped at ``subset_per_class``
    each), then for every indicator phrase replaces it with its neutral
    placeholder and re-runs the model to measure the shift in P(Male). Gender
    indicators are expected to move P(Male); region indicators test whether
    the Gender probe also leans on region cues. Returns the metadata dict.
    """
    run_dir, plot_dir = Path(run_dir), Path(plot_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, encoding="utf-8") as handle:
        dataset = json.load(handle)
    if "Gender" not in dataset["labels"]:
        raise ValueError(
            f"Ablation requires the 'Gender' attribute; dataset has labels for: {sorted(dataset['labels'])}"
        )

    all_labels = dataset["labels"]["Gender"]
    keep_indices = _balanced_keep_indices(all_labels, subset_per_class, seed)

    # Subset every parallel structure with the SAME ascending index list so
    # dataset row j <-> npz row j <-> baseline_probabilities row j.
    histories = [dataset["conversations_chat"][i] for i in keep_indices]
    labels = [all_labels[i] for i in keep_indices]
    history_ids = [dataset["history_ids"][i] for i in keep_indices]

    targets = load_targets([Path(p) for p in mapping_paths], control_words)
    indicators = [row for row in targets if row["target_type"].endswith("indicator")]
    row_targets: list[set[str]] = []
    for messages in histories:
        text = user_text(messages)
        values = matched_indicators(text, indicators)
        values.update(
            row["phrase"] for row in targets
            if not row["target_type"].endswith("indicator") and phrase_pattern(row["phrase"]).search(text)
        )
        row_targets.append(values)

    missing = [row["phrase"] for row in targets if not any(row["phrase"] in values for values in row_targets)]
    if missing:
        warnings.warn(
            f"{len(missing)} target(s) absent from the {len(histories)} selected histories "
            f"(they will report 0 ablations): {missing}",
            stacklevel=2,
        )
    presence_rows = save_presence_table(run_dir / "gender_indicator_presence.csv", targets, row_targets, labels)
    plot_presence(presence_rows, plot_dir / "gender_indicator_label_prevalence.png")

    artifact = joblib.load(run_dir / "best_probe_gender.joblib")
    hidden = np.load(run_dir / "hidden_states_personas.npz")[str(artifact.layer)][keep_indices]
    baseline_probabilities = artifact.classifier.predict_proba(artifact.scaler.transform(hidden))
    male_class = int(np.where(artifact.label_encoder.classes_ == "Male")[0][0])

    # Build ablated variants. history_index below is the LOCAL position within
    # the subset, so it indexes histories/labels/history_ids/baseline directly.
    variants: list[str] = []
    variant_meta: list[tuple[dict, int]] = []
    for target in targets:
        if target["target_type"].endswith("indicator"):
            replacement = f"<{target['indicator_name']}>"
        else:
            replacement = ""  # remove control words (no placeholder to restore)
        for history_index, messages in enumerate(histories):
            if target["phrase"] in row_targets[history_index]:
                variants.append(_format_messages(ablate_phrase(messages, target["phrase"], replacement)))
                variant_meta.append((target, history_index))

    if not variants:
        raise ValueError("No indicator or control phrase matched any selected history — nothing to ablate.")

    model_config = ModelConfig(model_name=str(model_name), device_map=device_map, torch_dtype=dtype)
    probe_config = ProbeConfig(layers=[artifact.layer], token_position="last")
    model, tokenizer = load_model_and_tokenizer(model_config)
    ablated_hidden = extract_hidden_states(
        model, tokenizer, variants, model_config, probe_config, batch_size=batch_size
    )[artifact.layer]
    ablated_probabilities = artifact.classifier.predict_proba(artifact.scaler.transform(ablated_hidden))

    effects: dict[str, list[float]] = defaultdict(list)
    detail_rows = []
    for row_index, (target, history_index) in enumerate(variant_meta):
        baseline = float(baseline_probabilities[history_index, male_class])
        ablated = float(ablated_probabilities[row_index, male_class])
        effect = baseline - ablated  # positive: phrase presence increases P(Male)
        effects[target["phrase"]].append(effect)
        detail_rows.append({
            "dataset_index": keep_indices[history_index],
            "history_id": history_ids[history_index],
            "true_label": labels[history_index],
            **target,
            "baseline_p_male": baseline,
            "ablated_p_male": ablated,
            "effect_of_phrase_presence_on_p_male": effect,
        })
    with (run_dir / "gender_phrase_ablation_details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    # Only summarize targets that actually got ablated (skip absent ones to
    # avoid np.mean([]) -> NaN warnings).
    summary_rows = []
    for target in targets:
        values = effects[target["phrase"]]
        if not values:
            continue
        summary_rows.append({
            **target,
            "histories_ablated": len(values),
            "mean_effect_on_p_male": float(np.mean(values)),
            "std_effect_on_p_male": float(np.std(values)),
            "median_effect_on_p_male": float(np.median(values)),
        })
    summary_rows.sort(key=lambda row: row["mean_effect_on_p_male"])
    with (run_dir / "gender_phrase_ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    def _bar_color(row: dict) -> str:
        if row["target_type"].startswith("Neutral"):
            return "#999999"
        return {"Male": "#d97706", "Female": "#2878b5"}.get(row["configured_gender"], "#59a14f")

    fig, ax = plt.subplots(figsize=(12, max(6, 0.32 * len(summary_rows))))
    y = np.arange(len(summary_rows))
    values = [row["mean_effect_on_p_male"] for row in summary_rows]
    errors = [row["std_effect_on_p_male"] / np.sqrt(row["histories_ablated"]) for row in summary_rows]
    colors = [_bar_color(row) for row in summary_rows]
    ax.barh(y, values, xerr=errors, color=colors, alpha=0.9, capsize=2)
    ax.set_yticks(y, [row["phrase"] for row in summary_rows])
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Mean effect of indicator presence on probe P(Male) (± SE)\n[indicator replaced with placeholder, e.g. 'action'→'<Movie>']")
    ax.set_title(f"Phrase ablation across {len(histories)} histories at layer {artifact.layer}")
    fig.tight_layout()
    fig.savefig(plot_dir / "gender_phrase_ablation_effect.png", dpi=160)
    plt.close(fig)

    plot_ablation_scatter(detail_rows, plot_dir / "ablation_scatter.png", artifact.layer)
    plot_effect_by_gender_group(detail_rows, plot_dir / "effect_by_gender_group.png", artifact.layer)

    per_dimension = Counter(row["dimension"] for row in targets)
    metadata = {
        "histories": len(histories),
        "label_counts": Counter(labels),
        "subset_per_class": subset_per_class,
        "total_available_before_balance": dict(Counter(l for l in all_labels if l is not None)),
        "probe_layer": int(artifact.layer),
        "indicator_count": len(indicators),
        "control_count": len(control_words),
        "targets_per_dimension": dict(per_dimension),
        "interventions": len(variants),
        "indicator_mappings": [str(Path(p).resolve()) for p in mapping_paths],
        "control_words": list(control_words),
        "effect_definition": "baseline P(Male) minus P(Male) after replacing an indicator with its placeholder (e.g. 'action' -> '<Movie>') or removing control words",
    }
    with (run_dir / "gender_phrase_ablation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=dict)
    print(json.dumps(metadata, indent=2, default=dict))
    return metadata


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    mapping_dir = repo / "src/generate_backgrounds/dimension_value_mapping"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(here / "data" / "dataset_personas.json"))
    parser.add_argument("--run-dir", default=str(here / "results" / "smollm2_gender_200"))
    parser.add_argument("--plot-dir", default=str(here / "plots" / "smollm2_gender_200"))
    parser.add_argument("--model", default=str(here / "models" / "SmolLM2-360M-Instruct"))
    parser.add_argument(
        "--mappings", nargs="+",
        default=[str(mapping_dir / "gender.csv"), str(mapping_dir / "region.csv")],
        help="Indicator mapping CSVs (gender.csv + region.csv by default)",
    )
    parser.add_argument("--control-words", nargs="+", default=DEFAULT_CONTROL_WORDS)
    parser.add_argument("--ablation-subset", type=int, default=0, help="Per-class cap (0 = all, still balanced)")
    parser.add_argument("--device-map", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_ablation(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        plot_dir=args.plot_dir,
        model_name=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        mapping_paths=args.mappings,
        control_words=args.control_words,
        subset_per_class=args.ablation_subset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
