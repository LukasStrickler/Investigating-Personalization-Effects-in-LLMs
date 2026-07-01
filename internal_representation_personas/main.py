#!/usr/bin/env python3
"""Run linear probes for attributes in the generated personas dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from collections import Counter

from config import PipelineConfig
from dataset import prepare_dataset

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probing pipeline for persona attributes")
    parser.add_argument("--personas-file", help="Path to personas.jsonl")
    parser.add_argument("--attributes", nargs="+", help="Persona fields to probe (default: Gender)")
    parser.add_argument("--include-partial", action="store_true", help="Include rows with missing persona fields")
    parser.add_argument("--samples", type=int, help="Maximum histories per joint persona group")
    parser.add_argument("--model", help="Hugging Face model name/path")
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse cached hidden states")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--token-position", default="last", choices=["last", "mean"])
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--results-dir")
    parser.add_argument("--plots-dir")
    parser.add_argument("--data-dir", help="Directory for the normalized dataset JSON")
    parser.add_argument(
        "--context-mode", default="full", choices=["full", "gender-turn-only"],
        help="Use full histories or isolate the generated Gender user turn",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    if args.personas_file:
        config.data.personas_file = args.personas_file
    if args.attributes:
        config.data.attributes = args.attributes
    config.data.include_partial = args.include_partial
    config.data.samples_per_group = args.samples
    config.data.context_mode = args.context_mode
    if args.model:
        config.model.model_name = args.model
    config.model.device_map = args.device_map
    config.model.torch_dtype = args.dtype
    config.model.max_seq_length = args.max_seq_length
    config.probe.token_position = args.token_position
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.plots_dir:
        config.plots_dir = args.plots_dir
    if args.data_dir:
        config.data.data_dir = args.data_dir
    return config


def _subset_hidden_states(hidden_states, indices):
    return {layer: values[indices] for layer, values in hidden_states.items()}


def _export_test_predictions(path, artifact, states, labels, source_indices, dataset):
    probabilities = artifact.classifier.predict_proba(artifact.scaler.transform(states))
    predictions = artifact.label_encoder.inverse_transform(probabilities.argmax(axis=1))
    classes = list(artifact.label_encoder.classes_)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fields = ["dataset_index", "history_id", "question", "true_label", "predicted_label", "confidence"]
        fields += [f"probability_{label}" for label in classes]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, local_index in enumerate(artifact.test_indices):
            dataset_index = source_indices[int(local_index)]
            item = {
                "dataset_index": dataset_index,
                "history_id": dataset["history_ids"][dataset_index],
                "question": dataset["last_user_questions"][dataset_index],
                "true_label": labels[int(local_index)],
                "predicted_label": predictions[row],
                "confidence": float(probabilities[row].max()),
            }
            item.update({f"probability_{label}": float(probabilities[row, i]) for i, label in enumerate(classes)})
            writer.writerow(item)


def main() -> None:
    args = parse_args()
    # Import heavyweight ML dependencies only after argument parsing so that
    # `python main.py --help` also works before requirements are installed.
    from extraction import extract_hidden_states, load_hidden_states, load_model_and_tokenizer, save_hidden_states
    from probing import train_probes
    from visualization import plot_layer_accuracy, plot_selectivity_gap

    config = build_config(args)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    print("=" * 70)
    print("  LINEAR PROBING PIPELINE — PERSONAS")
    print("=" * 70)
    print(f"  Model:           {config.model.model_name}")
    print(f"  Persona file:    {config.data.personas_file}")
    print(f"  Attributes:      {', '.join(config.data.attributes)}")
    print(f"  Histories/group: {config.data.samples_per_group or 'all'}")
    print(f"  Token position:  {config.probe.token_position}")
    print("=" * 70)
    started = time.time()

    print("\n▸ STEP 1: Loading personas dataset ...")
    dataset = prepare_dataset(config.data)
    conversations = dataset["conversations"]
    for attribute, labels in dataset["labels"].items():
        print(f"  {attribute}: {dict(Counter(value for value in labels if value is not None))}")

    hidden_path = os.path.join(config.results_dir, "hidden_states_personas.npz")
    if args.skip_extraction:
        if not os.path.exists(hidden_path):
            raise FileNotFoundError(f"No cached hidden states found at {hidden_path}")
        print("\n▸ STEP 2: Loading cached hidden states ...")
        hidden_states = load_hidden_states(hidden_path)
        cached_count = next(iter(hidden_states.values())).shape[0]
        if cached_count != len(conversations):
            raise ValueError(
                f"Cached hidden states contain {cached_count} rows, but the persona dataset contains "
                f"{len(conversations)}. Run again without --skip-extraction."
            )
    else:
        print("\n▸ STEP 2: Extracting hidden states ...")
        model, tokenizer = load_model_and_tokenizer(config.model)
        hidden_states = extract_hidden_states(model, tokenizer, conversations, config.model, config.probe)
        save_hidden_states(hidden_states, hidden_path)

    print("\n▸ STEP 3: Training persona probes + shuffled-label controls ...")
    summary: dict[str, dict] = {}
    for attribute in config.data.attributes:
        all_labels = dataset["labels"][attribute]
        indices = [index for index, label in enumerate(all_labels) if label is not None]
        labels = [all_labels[index] for index in indices]
        if len(set(labels)) < 2:
            print(f"  [WARN] Skipping {attribute}: fewer than two non-null classes")
            continue

        attribute_states = _subset_hidden_states(hidden_states, indices)
        results, artifacts = train_probes(
            attribute_states,
            labels,
            attribute,
            config.probe,
            test_size=config.data.test_size,
            seed=config.data.seed,
        )
        slug = attribute.lower().replace(" ", "_")
        results.save(os.path.join(config.results_dir, f"probe_{slug}.json"))
        plot_layer_accuracy(results, attribute, save_dir=config.plots_dir)
        plot_selectivity_gap(results, attribute, save_dir=config.plots_dir)

        best_real = results.best_layer(attribute, "logistic", control=False)
        best_control = results.best_layer(attribute, "logistic", control=True)
        if best_real and best_control:
            gap = best_real.accuracy - best_control.accuracy
            verdict = "ENCODED" if gap > 0.15 else "WEAK" if gap > 0.05 else "NOT FOUND"
            summary[attribute] = {
                "samples": len(labels),
                "classes": sorted(set(labels)),
                "best_layer": best_real.layer,
                "real_accuracy": round(best_real.accuracy, 4),
                "control_accuracy": round(best_control.accuracy, 4),
                "selectivity_gap": round(gap, 4),
                "verdict": verdict,
            }
            import joblib

            artifact = artifacts[best_real.layer]
            artifact.source_indices = __import__("numpy").asarray(indices)
            artifact_path = os.path.join(config.results_dir, f"best_probe_{slug}.joblib")
            joblib.dump(artifact, artifact_path)
            prediction_path = os.path.join(config.results_dir, f"test_predictions_{slug}.csv")
            _export_test_predictions(
                prediction_path,
                artifact,
                attribute_states[best_real.layer][artifact.test_indices],
                labels,
                indices,
                dataset,
            )
            summary[attribute]["probe_artifact"] = artifact_path
            summary[attribute]["test_predictions"] = prediction_path
            print(f"[Probing] Saved held-out predictions to {prediction_path}")

    summary_path = os.path.join(config.results_dir, "summary_personas.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"  Pipeline complete in {time.time() - started:.1f}s")
    print(f"  Results: {config.results_dir}")
    print(f"  Plots:   {config.plots_dir}")


if __name__ == "__main__":
    main()
