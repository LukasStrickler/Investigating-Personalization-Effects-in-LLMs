#!/usr/bin/env python3
"""Merge behavioral-audit stage2 judgment CSVs into one file per question.

The behavioral audit has multiple result folders for the same question. This
script concatenates the q1 files into one merged q1 CSV and the q2 files into
one merged q2 CSV so downstream analysis can operate on a single source per
question.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_AUDIT_DIR = (
    Path(__file__).resolve().parent.parent
)  # experiments/behavioral_audit (this file lives in significance_test/)
_RESULTS_ROOT = _AUDIT_DIR / "results_behavioral_audit"
_DEFAULT_SOURCE_DIRS = [
    _RESULTS_ROOT / "results_full001",
    _RESULTS_ROOT / "results_full001-e2b",
    _RESULTS_ROOT / "results_full001-ministral3-8b",
    _RESULTS_ROOT / "results_full002",
]
_DEFAULT_OUTPUT_DIR = _RESULTS_ROOT / "results_merged"


def _result_dir_to_run_tag(result_dir: Path) -> str:
    if not result_dir.name.startswith("results_"):
        raise ValueError(f"Unexpected results directory name: {result_dir}")
    return result_dir.name.removeprefix("results_")


def _discover_source_paths(question: str, source_dirs: list[Path]) -> list[Path]:
    source_paths: list[Path] = []
    for result_dir in source_dirs:
        if not result_dir.exists():
            continue
        run_tag = _result_dir_to_run_tag(result_dir)
        csv_path = result_dir / f"behavioral-audit-{run_tag}-{question}-stage2.judgments.csv"
        if csv_path.exists():
            source_paths.append(csv_path)
    return sorted(source_paths)


def _merge_csvs(source_paths: list[Path], output_path: Path) -> int:
    if not source_paths:
        raise FileNotFoundError(f"No stage2 judgment CSVs found for {output_path.stem}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] | None = None
    row_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer: csv.DictWriter[str] | None = None

        for source_path in source_paths:
            with source_path.open("r", encoding="utf-8", newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                if reader.fieldnames is None:
                    continue

                if fieldnames is None:
                    fieldnames = list(reader.fieldnames)
                    writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames) != fieldnames:
                    raise ValueError(
                        f"Schema mismatch while merging {source_path}:\n"
                        f"expected {fieldnames}\nfound    {list(reader.fieldnames)}"
                    )

                assert writer is not None
                for row in reader:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})
                    row_count += 1

    if fieldnames is None:
        raise ValueError(f"All source CSVs for {output_path} were empty")

    return row_count


def merge_stage2_judgments(
    output_dir: Path,
    source_dirs: list[Path] | None = None,
) -> list[Path]:
    source_dirs = source_dirs or _DEFAULT_SOURCE_DIRS

    outputs: list[Path] = []
    for question in ("q1", "q2"):
        source_paths = _discover_source_paths(question=question, source_dirs=source_dirs)
        output_path = output_dir / f"behavioral-audit-merged-{question}-stage2.judgments.csv"
        row_count = _merge_csvs(source_paths, output_path)
        print(f"Merged {len(source_paths)} files into {output_path} ({row_count} rows)")
        outputs.append(output_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge behavioral-audit stage2 judgments into one q1 file and one q2 file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where the merged q1/q2 CSVs will be written.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        dest="source_dirs",
        help="Additional results directory to include. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dirs = args.source_dirs or _DEFAULT_SOURCE_DIRS
    merge_stage2_judgments(output_dir=args.output_dir, source_dirs=source_dirs)


if __name__ == "__main__":
    main()
