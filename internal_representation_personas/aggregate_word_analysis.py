#!/usr/bin/env python3
"""Ablate every configured Gender indicator phrase plus neutral controls."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    """Replace or remove a phrase in all user turns.

    For Gender indicators, pass replacement=<IndicatorName> to restore the
    neutral template placeholder (e.g. 'action' -> '<Movie>'). For control
    words that have no placeholder, pass replacement="" to delete them.
    """
    result = deepcopy(messages)
    pattern = phrase_pattern(phrase)
    for message in result:
        if message.get("role") == "user":
            if replacement:
                message["content"] = pattern.sub(replacement, message["content"])
            else:
                message["content"] = re.sub(r"\s+([,.!?])", r"\1", pattern.sub("", message["content"]))
    return result


def load_targets(mapping_path: Path, controls: list[str]) -> list[dict[str, str]]:
    targets = []
    with mapping_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            targets.append({
                "phrase": row["Indicator_value"].strip(),
                "target_type": "Gender indicator",
                "configured_gender": row["Dimension_value"].strip(),
                "indicator_name": row["Indicator_name"].strip(),
            })
    indicator_phrases = {row["phrase"].casefold() for row in targets}
    for word in controls:
        if word.casefold() in indicator_phrases:
            raise ValueError(f"Control word is also a Gender indicator: {word}")
        targets.append({
            "phrase": word,
            "target_type": "Neutral template control",
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_presence(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row["target_type"] != "Gender indicator", row["configured_gender"], row["indicator_name"], row["phrase"]))
    y = np.arange(len(ordered))
    female = [float(row["p_female_given_present"]) for row in ordered]
    male = [float(row["p_male_given_present"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.barh(y, male, label="Male", color="#d97706")
    ax.barh(y, female, left=male, label="Female", color="#2878b5")
    ax.set_yticks(y, [row["phrase"] for row in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(true Gender label | phrase is present)")
    ax.set_title("Gender-label prevalence for 25 indicators and 10 neutral controls")
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
    color_map = {"Male": "#d97706", "Female": "#2878b5", "Control": "#999999"}
    label_map = {"Male": "Male indicator → ↑P(Male) when present",
                 "Female": "Female indicator → ↓P(Male) when present",
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
    groups: dict[str, list[float]] = {"Male": [], "Female": [], "Control": []}
    for row in detail_rows:
        key = row.get("configured_gender", "Control")
        if key in groups:
            groups[key].append(row["effect_of_phrase_presence_on_p_male"])

    order = [g for g in ("Male", "Female", "Control") if groups[g]]
    data = [groups[g] for g in order]
    color_map = {"Male": "#d97706", "Female": "#2878b5", "Control": "#999999"}
    xlabels = {
        "Male": "Male indicators\n(expect: positive)",
        "Female": "Female indicators\n(expect: negative)",
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


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(here / "data" / "dataset_personas.json"))
    parser.add_argument("--run-dir", default=str(here / "results" / "smollm2_gender_200"))
    parser.add_argument("--plot-dir", default=str(here / "plots" / "smollm2_gender_200"))
    parser.add_argument("--model", default=str(here / "models" / "SmolLM2-360M-Instruct"))
    parser.add_argument("--mapping", default=str(repo / "src/generate_backgrounds/dimension_value_mapping/gender.csv"))
    parser.add_argument("--control-words", nargs="+", default=DEFAULT_CONTROL_WORDS)
    parser.add_argument("--device-map", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_dir, plot_dir = Path(args.run_dir), Path(args.plot_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    with open(args.dataset, encoding="utf-8") as handle:
        dataset = json.load(handle)
    labels = dataset["labels"]["Gender"]
    histories = dataset["conversations_chat"]
    if Counter(labels) != Counter({"Female": 100, "Male": 100}):
        raise ValueError(f"Final analysis requires 100 Female and 100 Male histories; got {Counter(labels)}")

    targets = load_targets(Path(args.mapping), args.control_words)
    indicators = [row for row in targets if row["target_type"] == "Gender indicator"]
    row_targets: list[set[str]] = []
    for messages in histories:
        text = user_text(messages)
        values = matched_indicators(text, indicators)
        values.update(row["phrase"] for row in targets if row["target_type"] != "Gender indicator" and phrase_pattern(row["phrase"]).search(text))
        row_targets.append(values)

    missing = [row["phrase"] for row in targets if not any(row["phrase"] in values for values in row_targets)]
    if missing:
        raise ValueError(f"Targets absent from selected 200 histories: {missing}")
    presence_rows = save_presence_table(run_dir / "gender_indicator_presence.csv", targets, row_targets, labels)
    plot_presence(presence_rows, plot_dir / "gender_indicator_label_prevalence.png")

    artifact = joblib.load(run_dir / "best_probe_gender.joblib")
    hidden = np.load(run_dir / "hidden_states_personas.npz")[str(artifact.layer)]
    baseline_probabilities = artifact.classifier.predict_proba(artifact.scaler.transform(hidden))
    male_class = int(np.where(artifact.label_encoder.classes_ == "Male")[0][0])

    variants: list[str] = []
    variant_meta: list[tuple[dict, int]] = []
    for target in targets:
        if target["target_type"] == "Gender indicator":
            replacement = f"<{target['indicator_name']}>"
        else:
            replacement = ""  # remove control words (no placeholder to restore)
        for index, messages in enumerate(histories):
            if target["phrase"] in row_targets[index]:
                variants.append(_format_messages(ablate_phrase(messages, target["phrase"], replacement)))
                variant_meta.append((target, index))

    model_config = ModelConfig(model_name=args.model, device_map=args.device_map, torch_dtype=args.dtype)
    probe_config = ProbeConfig(layers=[artifact.layer], token_position="last")
    model, tokenizer = load_model_and_tokenizer(model_config)
    ablated_hidden = extract_hidden_states(
        model, tokenizer, variants, model_config, probe_config, batch_size=args.batch_size
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
            "dataset_index": history_index,
            "history_id": dataset["history_ids"][history_index],
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

    summary_rows = []
    for target in targets:
        values = effects[target["phrase"]]
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

    fig, ax = plt.subplots(figsize=(12, 13))
    y = np.arange(len(summary_rows))
    values = [row["mean_effect_on_p_male"] for row in summary_rows]
    errors = [row["std_effect_on_p_male"] / np.sqrt(row["histories_ablated"]) for row in summary_rows]
    colors = ["#999999" if row["target_type"].startswith("Neutral") else ("#d97706" if row["configured_gender"] == "Male" else "#2878b5") for row in summary_rows]
    ax.barh(y, values, xerr=errors, color=colors, alpha=0.9, capsize=2)
    ax.set_yticks(y, [row["phrase"] for row in summary_rows])
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Mean effect of indicator presence on probe P(Male) (± SE)\n[indicator replaced with placeholder, e.g. 'action'→'<Movie>']")
    ax.set_title(f"Phrase ablation across 200 histories at layer {artifact.layer}")
    fig.tight_layout()
    fig.savefig(plot_dir / "gender_phrase_ablation_effect.png", dpi=160)
    plt.close(fig)

    plot_ablation_scatter(detail_rows, plot_dir / "ablation_scatter.png", artifact.layer)
    plot_effect_by_gender_group(detail_rows, plot_dir / "effect_by_gender_group.png", artifact.layer)

    metadata = {
        "histories": len(histories),
        "label_counts": Counter(labels),
        "probe_layer": int(artifact.layer),
        "indicator_count": len(indicators),
        "control_count": len(args.control_words),
        "interventions": len(variants),
        "indicator_mapping": str(Path(args.mapping).resolve()),
        "control_words": args.control_words,
        "effect_definition": "baseline P(Male) minus P(Male) after replacing Gender indicator with its placeholder (e.g. 'action' -> '<Movie>') or removing control words",
    }
    with (run_dir / "gender_phrase_ablation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=dict)
    print(json.dumps(metadata, indent=2, default=dict))


if __name__ == "__main__":
    main()
