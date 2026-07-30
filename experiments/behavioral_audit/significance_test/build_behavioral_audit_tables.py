#!/usr/bin/env python3
"""Run the full behavioral-audit table pipeline.

This wrapper merges the stage2 judgment CSVs into one q1 file and one q2 file,
builds the frequency tables from those merged inputs, and then runs the
significance analysis on the resulting frequency tables.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_AUDIT_DIR = (
    Path(__file__).resolve().parent.parent
)  # experiments/behavioral_audit (this file lives in significance_test/)
_RESULTS_ROOT = _AUDIT_DIR / "results_behavioral_audit"
_DEFAULT_OUTPUT_ROOT = _RESULTS_ROOT / "results_merged"

if str(_AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(_AUDIT_DIR))

build_frequency_tables = importlib.import_module("build_frequency_tables").build_frequency_tables
analyze_frequency_tables = importlib.import_module(
    "build_significance_tables"
).analyze_frequency_tables
merge_stage2_judgments = importlib.import_module("merge_stage2_judgments").merge_stage2_judgments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge behavioral-audit stage2 judgments and build frequency/significance tables."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help="Directory where merged outputs and derived tables will be written.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        dest="source_dirs",
        help="Additional results directory to include when merging. Can be passed multiple times.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Number of permutations for the overall significance test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the significance test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dirs = args.source_dirs

    print("[1/3] Merging stage2 judgment CSVs...", flush=True)
    merged_paths = merge_stage2_judgments(output_dir=args.output_root, source_dirs=source_dirs)
    q1_csv, q2_csv = merged_paths

    frequency_dir = args.output_root / "frequency_tables"
    print("[2/3] Building frequency tables...", flush=True)
    frequency_outputs = build_frequency_tables(
        q1_csv=q1_csv, q2_csv=q2_csv, output_dir=frequency_dir
    )

    print("[3/3] Building significance tables...", flush=True)
    significance_outputs = analyze_frequency_tables(
        input_dir=frequency_dir,
        output_dir=frequency_dir / "significance_tests",
        permutations=args.permutations,
        seed=args.seed,
    )

    print("Completed behavioral-audit table pipeline:")
    print(f"- merged q1: {q1_csv}")
    print(f"- merged q2: {q2_csv}")
    print(f"- frequency tables: {len(frequency_outputs)} files")
    print(f"- significance tables: {len(significance_outputs)} files")


if __name__ == "__main__":
    main()
