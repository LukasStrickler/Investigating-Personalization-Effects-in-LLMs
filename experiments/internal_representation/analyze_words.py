#!/usr/bin/env python3
"""Estimate word importance by deleting user words and measuring probe-score changes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ModelConfig, ProbeConfig
from dataset import _format_messages
from extraction import extract_hidden_states, load_model_and_tokenizer


def _candidate_words(messages, limit: int) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][\w'-]+", message.get("content", "")):
            key = word.casefold()
            if key not in seen:
                seen.add(key)
                words.append(word)
            if len(words) >= limit:
                return words
    return words


def _without_word(messages, word: str):
    result = deepcopy(messages)
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    for message in result:
        if message.get("role") == "user":
            message["content"] = pattern.sub("", message["content"])
    return result


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Word ablation for the best Gender probe")
    parser.add_argument("--dataset", default=str(here / "data" / "dataset_personas.json"))
    parser.add_argument("--probe", default=str(here / "results" / "best_probe_gender.joblib"))
    parser.add_argument(
        "--model",
        default=str(here / "models" / "SmolLM2-360M-Instruct"),
        help="HF ID or downloaded local model path",
    )
    parser.add_argument("--dataset-index", type=int, help="Row from test_predictions_gender.csv")
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as handle:
        dataset = json.load(handle)
    artifact = joblib.load(args.probe)
    if args.dataset_index is None:
        local_index = int(artifact.test_indices[0])
        index = (
            int(artifact.source_indices[local_index])
            if artifact.source_indices is not None
            else local_index
        )
    else:
        index = args.dataset_index

    messages = dataset["conversations_chat"][index]
    words = _candidate_words(messages, args.max_words)
    variants = [messages] + [_without_word(messages, word) for word in words]
    texts = [_format_messages(item) for item in variants]

    model_config = ModelConfig(
        model_name=args.model, device_map=args.device_map, torch_dtype=args.dtype
    )
    probe_config = ProbeConfig(layers=[artifact.layer], token_position="last")
    model, tokenizer = load_model_and_tokenizer(model_config)
    states = extract_hidden_states(model, tokenizer, texts, model_config, probe_config)[
        artifact.layer
    ]
    probabilities = artifact.classifier.predict_proba(artifact.scaler.transform(states))
    baseline_class = int(probabilities[0].argmax())
    baseline_score = float(probabilities[0, baseline_class])
    predicted_label = str(artifact.label_encoder.inverse_transform([baseline_class])[0])

    rows = [
        {"word": word, "importance": baseline_score - float(probabilities[i + 1, baseline_class])}
        for i, word in enumerate(words)
    ]
    rows.sort(key=lambda row: abs(row["importance"]), reverse=True)
    output_csv = here / "results" / f"word_importance_{index}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "importance"])
        writer.writeheader()
        writer.writerows(rows)

    shown = rows[: args.top][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(shown) * 0.32)))
    ax.barh([row["word"] for row in shown], [row["importance"] for row in shown])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Drop in P({predicted_label}) when word is removed")
    ax.set_title(f"Word ablation — dataset row {index}, layer {artifact.layer}")
    fig.tight_layout()
    output_plot = here / "plots" / f"word_importance_{index}.png"
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=150)
    print(f"Baseline prediction: {predicted_label} ({baseline_score:.3f})")
    print(f"Saved {output_csv}")
    print(f"Saved {output_plot}")


if __name__ == "__main__":
    main()
