#!/usr/bin/env python3
"""
Linear Probing Pipeline — Main Entry Point.

Orchestrates the pipeline:
  1. Generate / load the labeled conversation dataset
  2. Load the frozen LLM and extract hidden states
  3. Train linear probes (real + control) per layer
  4. Evaluate selectivity and generate plots

Usage:
    python main.py                              # Full pipeline with defaults
    python main.py --skip-extraction            # Reuse cached hidden states
    python main.py --model mistralai/Mistral-7B-Instruct-v0.3
    python main.py --samples 100               # More samples per class
    python main.py --data-file /path/to/data.json  # Use your own labeled data
"""

import argparse
import json
import os
import time
import warnings
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import DataConfig, ModelConfig, PipelineConfig, ProbeConfig
from dataset import prepare_dataset
from extraction import (
    extract_hidden_states,
    load_hidden_states,
    load_model_and_tokenizer,
    save_hidden_states,
)
from probing import ProbeResults, train_probes
from visualization import plot_layer_accuracy, plot_selectivity_gap

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear Probing Pipeline")
    p.add_argument("--model", type=str, default=None, help="HuggingFace model name/path")
    p.add_argument("--samples", type=int, default=None, help="Samples per expertise level")
    p.add_argument("--data-file", type=str, default=None,
                   help="Path to a pre-labeled JSON dataset (skips generation). "
                        "Schema: {conversations: [...], labels: {<label_key>: [...]}}")
    p.add_argument("--label-key", type=str, default="expertise",
                   help="Which label to probe (must exist in labels dict, e.g. 'gender', 'expertise')")
    p.add_argument("--skip-extraction", action="store_true",
                   help="Skip extraction; load cached hidden states from disk")
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--token-position", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--eval-mode", type=str, default="repeated_kfold",
                   choices=["split", "repeated_kfold"],
                   help="Evaluation mode: 'split' = single train/test split, "
                        "'repeated_kfold' = repeated stratified k-fold (more robust)")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--plots-dir", type=str, default="plots")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if args.model:
        cfg.model.model_name = args.model
    if args.samples:
        cfg.data.samples_per_class = args.samples
    cfg.model.device_map = args.device_map
    cfg.model.torch_dtype = args.dtype
    cfg.model.max_seq_length = args.max_seq_length
    cfg.probe.token_position = args.token_position
    cfg.probe.eval_mode = args.eval_mode
    cfg.results_dir = args.results_dir
    cfg.plots_dir = args.plots_dir
    return cfg


def load_data_file(path: str, label_key: str = "expertise") -> dict:
    """Load a pre-labeled JSON dataset from disk."""
    with open(path) as f:
        data = json.load(f)
    assert "conversations" in data and "labels" in data and label_key in data["labels"], (
        f"--data-file must contain 'conversations' and 'labels.{label_key}' keys. "
        f"Got keys: {list(data.keys())}, label keys: {list(data.get('labels', {}).keys())}"
    )
    return data


def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)

    label_key = args.label_key

    print("=" * 70)
    print(f"  LINEAR PROBING PIPELINE — {label_key.upper()}")
    print("=" * 70)
    print(f"  Model:           {cfg.model.model_name}")
    print(f"  Device map:      {cfg.model.device_map}")
    print(f"  Dtype:           {cfg.model.torch_dtype}")
    print(f"  Label key:       {label_key}")
    print(f"  Token position:  {cfg.probe.token_position}")
    print("=" * 70)

    t0 = time.time()

    # ── Step 1: Dataset ────────────────────────────────────────────────
    print("\n▸ STEP 1: Preparing dataset ...")
    if args.data_file:
        print(f"  Loading from {args.data_file}")
        dataset = load_data_file(args.data_file, label_key)
    else:
        dataset = prepare_dataset(cfg.data)

    conversations = dataset["conversations"]
    labels = dataset["labels"][label_key]
    print(f"  Total conversations: {len(conversations)}")
    print(f"  Class distribution: {dict(Counter(labels))}")

    # ── Step 2: Hidden State Extraction ────────────────────────────────
    hs_path = os.path.join(cfg.results_dir, "hidden_states.npz")

    if args.skip_extraction and os.path.exists(hs_path):
        print("\n▸ STEP 2: Loading cached hidden states ...")
        hidden_states = load_hidden_states(hs_path)
    else:
        print("\n▸ STEP 2: Extracting hidden states ...")
        model, tokenizer = load_model_and_tokenizer(cfg.model)
        hidden_states = extract_hidden_states(
            model, tokenizer, conversations, cfg.model, cfg.probe
        )
        save_hidden_states(hidden_states, hs_path)

    t1 = time.time()
    print(f"  Extraction took {t1 - t0:.1f}s")

    # ── Step 3: Train Probes ────────────────────────────────────────────
    print("\n▸ STEP 3: Training linear probes + control tasks ...")
    le = LabelEncoder()
    le.fit(labels)

    probe_results, _ = train_probes(
        hidden_states, labels, label_key, cfg.probe,
        test_size=cfg.data.test_size, seed=cfg.data.seed,
    )
    probe_results.save(os.path.join(cfg.results_dir, f"probe_{label_key}.json"))

    t2 = time.time()
    print(f"  Probing took {t2 - t1:.1f}s")

    # ── Step 4: Visualize + Summary ────────────────────────────────────
    print("\n▸ STEP 4: Generating plots ...")
    plot_layer_accuracy(probe_results, label_key, save_dir=cfg.plots_dir)
    plot_selectivity_gap(probe_results, label_key, save_dir=cfg.plots_dir)

    best_real = probe_results.best_layer(label_key, "logistic", control=False)
    best_ctrl = probe_results.best_layer(label_key, "logistic", control=True)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    summary = {}
    if best_real and best_ctrl:
        gap = best_real.accuracy - best_ctrl.accuracy
        verdict = "✓ ENCODED" if gap > 0.15 else "? WEAK" if gap > 0.05 else "✗ NOT FOUND"
        summary[label_key] = {
            "best_layer": best_real.layer,
            "real_accuracy": round(best_real.accuracy, 4),
            "control_accuracy": round(best_ctrl.accuracy, 4),
            "selectivity_gap": round(gap, 4),
            "verdict": verdict,
        }
        print(f"  {label_key}  layer={best_real.layer:3d}  "
              f"acc={best_real.accuracy:.3f}  ctrl={best_ctrl.accuracy:.3f}  "
              f"gap={gap:+.3f}  {verdict}")

    with open(os.path.join(cfg.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    t3 = time.time()
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {t3 - t0:.1f}s")
    print(f"  Results:  {cfg.results_dir}/")
    print(f"  Plots:    {cfg.plots_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
