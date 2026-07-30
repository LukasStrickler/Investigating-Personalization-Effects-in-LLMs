#!/usr/bin/env python3
"""Build frequency tables for merged behavioral audit stage2 judgments.

Creates four tables:
1) q1 x Gender x profession-class frequency
2) q1 x Race x profession-class frequency
3) q2 x Gender x profession-class frequency
4) q2 x Race x profession-class frequency

Each table is written as CSV with row/column totals.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def _normalize(value: str | None, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _update_count_table(
    table: dict[str, dict[str, int]],
    row_key: str,
    col_key: str,
) -> None:
    if row_key not in table:
        table[row_key] = defaultdict(int)
    table[row_key][col_key] += 1


def _load_count_table(
    csv_path: Path,
    attribute_key: str,
    model_alias: str | None = None,
) -> dict[str, dict[str, int]]:
    """Load one stage2 judgments CSV into a frequency table.

    Rows are values of attribute_key (e.g., true_gender or true_race).
    Columns are the model output classes from final_class.
    """
    table: dict[str, dict[str, int]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_model_alias = _normalize(row.get("subject_model_alias"), fallback="UnknownModel")
            if model_alias is not None and row_model_alias != model_alias:
                continue

            status = _normalize(row.get("status"), fallback="")
            parse_status = _normalize(row.get("parse_status"), fallback="")
            if status.lower() != "success" or parse_status.lower() != "matched":
                continue

            final_class = _normalize(row.get("final_class"))
            try:
                metadata = json.loads(row.get("metadata") or "{}")
            except json.JSONDecodeError:
                metadata = {}

            attr_value = _normalize(metadata.get(attribute_key))
            _update_count_table(table, row_key=attr_value, col_key=final_class)

    return table


def _write_frequency_table(table: dict[str, dict[str, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_labels = sorted(table.keys())
    col_labels: set[str] = set()
    for row_counts in table.values():
        col_labels.update(row_counts.keys())
    col_labels_sorted = sorted(col_labels)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attribute_value", *col_labels_sorted, "Total"])
        for row_label in row_labels:
            counts = table[row_label]
            row_total = sum(counts.get(col, 0) for col in col_labels_sorted)
            writer.writerow([row_label, *[counts.get(col, 0) for col in col_labels_sorted], row_total])

        column_totals = [sum(table[row_label].get(col, 0) for row_label in row_labels) for col in col_labels_sorted]
        grand_total = sum(column_totals)
        writer.writerow(["Total", *column_totals, grand_total])


def _safe_folder_name(model_alias: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", model_alias.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "UnknownModel"


def _collect_model_aliases(*csv_paths: Path) -> list[str]:
    aliases: set[str] = set()
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                aliases.add(_normalize(row.get("subject_model_alias"), fallback="UnknownModel"))
    return sorted(aliases)


def _write_tables_for_specs(
    specs: list[tuple[str, Path, str, str]],
    output_dir: Path,
    model_alias: str | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    for _, input_path, attribute_key, filename in specs:
        table = _load_count_table(input_path, attribute_key=attribute_key, model_alias=model_alias)
        csv_path = output_dir / filename
        _write_frequency_table(table, csv_path)
        outputs.append(csv_path)
    return outputs


def build_frequency_tables(q1_csv: Path, q2_csv: Path, output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    specs = [
        ("q1", q1_csv, "true_gender", "q1_gender_frequency.csv"),
        ("q1", q1_csv, "true_race", "q1_race_frequency.csv"),
        ("q2", q2_csv, "true_gender", "q2_gender_frequency.csv"),
        ("q2", q2_csv, "true_race", "q2_race_frequency.csv"),
    ]

    # Overall (all models combined)
    outputs.extend(_write_tables_for_specs(specs, output_dir=output_dir, model_alias=None))

    # Per-model tables
    model_aliases = _collect_model_aliases(q1_csv, q2_csv)
    for model_alias in model_aliases:
        model_dir = output_dir / "by_model" / _safe_folder_name(model_alias)
        outputs.extend(_write_tables_for_specs(specs, output_dir=model_dir, model_alias=model_alias))

    # Record folder-to-model mapping for traceability
    mapping_path = output_dir / "by_model" / "model_folder_mapping.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_alias", "folder_name"])
        for model_alias in model_aliases:
            writer.writerow([model_alias, _safe_folder_name(model_alias)])
    outputs.append(mapping_path)

    return outputs


def parse_args() -> argparse.Namespace:
    default_root = Path("experiments/behavioral_audit/results_merged")
    parser = argparse.ArgumentParser(
        description="Create q1/q2 gender/race frequency tables mapped to final_class values from merged stage2 judgments."
    )
    parser.add_argument(
        "--q1-csv",
        type=Path,
        default=default_root / "behavioral-audit-merged-q1-stage2.judgments.csv",
        help="Path to the merged q1 stage2 judgments CSV.",
    )
    parser.add_argument(
        "--q2-csv",
        type=Path,
        default=default_root / "behavioral-audit-merged-q2-stage2.judgments.csv",
        help="Path to the merged q2 stage2 judgments CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "frequency_tables",
        help="Directory where frequency table CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_frequency_tables(args.q1_csv, args.q2_csv, args.output_dir)
    print("Wrote frequency tables:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
