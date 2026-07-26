#!/usr/bin/env python3
"""Build significance tables from behavioral audit frequency tables.

This script reads the CSV frequency tables produced by
build_frequency_tables.py and writes two readable outputs per table:

1) an overall independence test for the full contingency table
2) one-vs-rest 2x2 tests for each profession category

The per-category output is intentionally table-shaped so it is easy to read:
each row contains the 2x2 counts, row/column totals, the test result, and
multiple-testing correction.
"""

from __future__ import annotations
from mpmath import mp, mpf, binomial as mpcomb

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ContingencyResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float

def _read_frequency_table(csv_path: Path) -> tuple[list[str], list[str], list[list[int]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

        if not header or header[0] != "attribute_value":
            raise ValueError(f"Unexpected table format in {csv_path}")

        col_labels = [label for label in header[1:] if label != "Total"]
        row_labels: list[str] = []
        rows: list[list[int]] = []

        for row in reader:
            if not row:
                continue
            row_label = row[0]
            if row_label == "Total" or row_label.strip().lower() == "unknown":
                continue
            values = [int(value) for value in row[1 : 1 + len(col_labels)]]

            row_labels.append(row_label)
            rows.append(values)

    return row_labels, col_labels, rows


def _expected_counts(table: list[list[int]]) -> list[list[float]]:
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[row_idx][col_idx] for row_idx in range(len(table))) for col_idx in range(len(table[0]))]
    grand_total = sum(row_totals)

    expected: list[list[float]] = []
    for row_total in row_totals:
        expected.append([(row_total * col_total) / grand_total for col_total in col_totals])
    return expected


def _chi_square_statistic(table: list[list[int]]) -> float:
    expected = _expected_counts(table)
    stat = 0.0
    for row_idx, row in enumerate(table):
        for col_idx, observed in enumerate(row):
            exp = expected[row_idx][col_idx]
            if exp > 0:
                stat += (observed - exp) ** 2 / exp
    return stat


def _cramers_v(table: list[list[int]]) -> float:
    chi2 = _chi_square_statistic(table)
    n = sum(sum(row) for row in table)
    if n == 0:
        return 0.0
    rows = len(table)
    cols = len(table[0]) if table else 0
    denom = min(rows - 1, cols - 1)
    if denom <= 0:
        return 0.0
    return math.sqrt(chi2 / (n * denom))


def _shuffle_p_value(table: list[list[int]], permutations: int, seed: int) -> float:
    """Permutation p-value for a full contingency table.

    We expand the table to paired row/column labels, shuffle the row labels,
    and recompute the chi-square statistic.
    """
    rng = random.Random(seed)
    row_labels: list[int] = []
    col_labels: list[int] = []
    for row_idx, row in enumerate(table):
        for col_idx, count in enumerate(row):
            row_labels.extend([row_idx] * count)
            col_labels.extend([col_idx] * count)

    observed = _chi_square_statistic(table)
    hits = 0

    for _ in range(permutations):
        shuffled = row_labels[:]
        rng.shuffle(shuffled)

        perm_table = [[0 for _ in range(len(table[0]))] for _ in range(len(table))]
        for row_idx, col_idx in zip(shuffled, col_labels):
            perm_table[row_idx][col_idx] += 1

        if _chi_square_statistic(perm_table) >= observed:
            hits += 1

    return (hits + 1) / (permutations + 1)


def _fisher_exact_two_sided(table: list[list[int]], precision: int = 300) -> tuple[float, float]:
    """Return odds ratio and two-sided Fisher exact p-value for a 2x2 table.
    
    Uses mpmath for arbitrary-precision arithmetic to avoid floating-point
    underflow for very extreme p-values (e.g. p < 10^-300).
    """
    a, b = table[0]
    c, d = table[1]

    if b * c == 0:
        odds_ratio = float("inf") if a * d > 0 else 0.0
    else:
        odds_ratio = (a * d) / (b * c)

    row_total = a + b
    col_total = a + c
    n = a + b + c + d

    mp.dps = precision  # decimal places of precision

    def hypergeom_prob(x: int) -> mpf:
        # mpmath comb returns exact integer, division gives high-precision float
        return mpcomb(col_total, x) * mpcomb(n - col_total, row_total - x) / mpcomb(n, row_total)

    min_x = max(0, row_total - (n - col_total))
    max_x = min(row_total, col_total)
    observed = hypergeom_prob(a)

    p_value = mpf(0)
    for x in range(min_x, max_x + 1):
        prob = hypergeom_prob(x)
        if prob <= observed * (1 + mpf("1e-10")):  # small tolerance for floating comparison
            p_value += prob

    # Convert back to Python float; extremely small values stay representable
    # as mpf but are stored as float in CSV — clamp to float min if needed
    p_float = float(min(p_value, mpf(1)))
    return odds_ratio, max(p_float, 0.0)


def _one_vs_rest_table(table: list[list[int]], row_idx: int, col_idx: int) -> list[list[int]]:
    target = table[row_idx][col_idx]
    target_outside = sum(table[row_idx]) - target
    rest_in = sum(table[r][col_idx] for r in range(len(table)) if r != row_idx)
    rest_outside = sum(sum(table[r]) for r in range(len(table)) if r != row_idx) - rest_in
    return [[target, target_outside], [rest_in, rest_outside]]


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []

    n = len(p_values)
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * n
    running_min = 1.0

    for rank, (idx, p_value) in enumerate(reversed(ranked), start=1):
        order = n - rank + 1
        adjusted_value = min(running_min, (p_value * n) / order)
        running_min = adjusted_value
        adjusted[idx] = min(adjusted_value, 1.0)

    return adjusted


def _load_frequency_tables(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*_frequency.csv")
        if path.name != "model_folder_mapping.csv"
    )


def _table_metadata_from_path(path: Path) -> tuple[str, str | None, str]:
    parts = path.parts
    model_alias: str | None = None
    if "by_model" in parts:
        idx = parts.index("by_model")
        model_alias = parts[idx + 1]

    stem = path.stem
    if stem.startswith("q1_"):
        question_key = "q1"
    elif stem.startswith("q2_"):
        question_key = "q2"
    else:
        question_key = stem

    if "gender" in stem:
        attribute = "Gender"
    elif "race" in stem:
        attribute = "Race"
    else:
        attribute = "Unknown"

    return question_key, model_alias, attribute


def _overall_results(table: list[list[int]], permutations: int, seed: int) -> ContingencyResult:
    chi2 = _chi_square_statistic(table)
    p_value = _shuffle_p_value(table, permutations=permutations, seed=seed)
    effect_size = _cramers_v(table)
    return ContingencyResult(
        test_name="permutation_chi_square",
        statistic=chi2,
        p_value=p_value,
        effect_size=effect_size,
    )


def _category_results(
    row_labels: list[str],
    col_labels: list[str],
    table: list[list[int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            two_by_two = _one_vs_rest_table(table, row_idx, col_idx)
            odds_ratio, p_value = _fisher_exact_two_sided(two_by_two)

            a, b = two_by_two[0]
            c, d = two_by_two[1]
            rows.append(
                {
                    "category_value": col_label,
                    "group_value": row_label,
                    "group_in_category": a,
                    "group_outside_category": b,
                    "other_in_category": c,
                    "other_outside_category": d,
                    "group_total": a + b,
                    "other_total": c + d,
                    "category_total": a + c,
                    "non_category_total": b + d,
                    "odds_ratio": odds_ratio,
                    "p_value": p_value,
                    "statistic": float("nan"),
                    "test_name": "fisher_exact_2x2",
                }
            )

    corrected = _benjamini_hochberg([float(str(row["p_value"])) for row in rows])
    for row, corrected_p in zip(rows, corrected):
        row["corrected_p_value"] = corrected_p
        row["reject_fdr"] = corrected_p < 0.05

    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _progress_bar(current: int, total: int, label: str) -> None:
    width = 24
    filled = width if total == 0 else int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 if total == 0 else 100.0 * current / total
    sys.stderr.write(f"\r{label} [{bar}] {current}/{total} ({percent:.0f}%)")
    sys.stderr.flush()


def analyze_frequency_tables(
    input_dir: Path,
    output_dir: Path,
    permutations: int,
    seed: int,
) -> list[Path]:
    outputs: list[Path] = []
    table_paths = _load_frequency_tables(input_dir)
    total_tables = len(table_paths)
    for index, table_path in enumerate(table_paths, start=1):
        _progress_bar(index, total_tables, "Significance tables")
        question_key, model_alias, attribute = _table_metadata_from_path(table_path)
        row_labels, col_labels, table = _read_frequency_table(table_path)

        if len(table) < 2 or len(col_labels) < 2:
            print(f"[SKIP] {table_path} — degenerate table ({len(table)} rows, {len(col_labels)} cols)")
            continue

        table_slug = table_path.stem.replace("_frequency", "")
        target_dir = output_dir / (model_alias or "all_models")

        overall = _overall_results(table, permutations=permutations, seed=seed)
        overall_path = target_dir / f"{table_slug}__overall.csv"
        _write_csv(
            overall_path,
            [
                {
                    "question_key": question_key,
                    "model_alias": model_alias or "all_models",
                    "attribute": attribute,
                    "test_name": overall.test_name,
                    "statistic": overall.statistic,
                    "p_value": overall.p_value,
                    "effect_size": overall.effect_size,
                    "reject_fdr": overall.p_value < 0.05,
                    "n_rows": len(row_labels),
                    "n_cols": len(col_labels),
                    "grand_total": sum(sum(row) for row in table),
                }
            ],
            [
                "question_key",
                "model_alias",
                "attribute",
                "test_name",
                "statistic",
                "p_value",
                "effect_size",
                "reject_fdr",
                "n_rows",
                "n_cols",
                "grand_total",
            ],
        )
        outputs.append(overall_path)

        category_rows = _category_results(row_labels=row_labels, col_labels=col_labels, table=table)
        category_path = target_dir / f"{table_slug}__category_tests.csv"
        _write_csv(
            category_path,
            (
                {
                    "question_key": question_key,
                    "model_alias": model_alias or "all_models",
                    "attribute": attribute,
                    **row,
                }
                for row in category_rows
            ),
            [
                "question_key",
                "model_alias",
                "attribute",
                "category_value",
                "group_value",
                "group_in_category",
                "group_outside_category",
                "other_in_category",
                "other_outside_category",
                "group_total",
                "other_total",
                "category_total",
                "non_category_total",
                "odds_ratio",
                "p_value",
                "corrected_p_value",
                "reject_fdr",
                "statistic",
                "test_name",
            ],
        )
        outputs.append(category_path)

    if total_tables > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()

    return outputs


def parse_args() -> argparse.Namespace:
    default_root = Path("experiments/behavioral_audit/results_merged/frequency_tables")
    parser = argparse.ArgumentParser(
        description="Test significance from frequency tables built by the behavioral audit pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_root,
        help="Directory containing frequency table CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "significance_tests",
        help="Directory where significance CSVs will be written.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Number of permutations for the overall independence test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the permutation test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = analyze_frequency_tables(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        permutations=args.permutations,
        seed=args.seed,
    )
    print("Wrote significance tables:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
