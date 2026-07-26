#!/usr/bin/env python3
"""Build LaTeX tables from behavioral-audit significance test outputs.

This exporter creates:
1) two compact summary tables for the all-models results (gender and region)
2) detailed appendix tables for all models combined and each individual model,
   split by gender and region

The main summary tables keep only the top 8 rows per attribute. Rows are ranked
by corrected p-value, then by effect strength, then by category support.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_AUDIT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT_DIR = _AUDIT_DIR / "results_merged" / "frequency_tables" / "significance_tests"
_DEFAULT_OUTPUT_DIR = _AUDIT_DIR / "results_merged" / "latex_tables"
_MODEL_ORDER = ["all_models", "deepseek-v4-flash_paid", "gemma-4-31b_paid", "gemma-4-e2b_modal", "glm-5.2_paid", "grok-4.3_paid", "ministral-3-8b_modal"]
_SUMMARY_WIDTHS = {
    "category": "0.26\\linewidth",
    "class": "0.14\\linewidth",
    "ncat": "0.07\\linewidth",
    "or": "0.08\\linewidth",
    "padj": "0.20\\linewidth",
}
_APPENDIX_WIDTHS = {
    "category": "0.35\\linewidth",
    "class": "0.1\\linewidth",
    "a": "0.05\\linewidth",
    "b": "0.05\\linewidth",
    "c": "0.05\\linewidth",
    "d": "0.05\\linewidth",
    "ngroup": "0.05\\linewidth",
    "nother": "0.05\\linewidth",
    "ncat": "0.05\\linewidth",
    "or": "0.05\\linewidth",
    "p": "0.05\\linewidth",
    "padj": "0.06\\linewidth",
}
@dataclass(frozen=True)
class DetailedRow:
    question_key: str
    model_alias: str
    attribute: str
    category_value: str
    group_value: str
    group_in_category: int
    group_outside_category: int
    other_in_category: int
    other_outside_category: int
    group_total: int
    other_total: int
    category_total: int
    non_category_total: int
    odds_ratio: float
    p_value: float
    corrected_p_value: float
    reject_fdr: bool
    statistic: float | None
    test_name: str


def _read_rows(path: Path) -> list[DetailedRow]:
    rows: list[DetailedRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                DetailedRow(
                    question_key=row["question_key"],
                    model_alias=row["model_alias"],
                    attribute=row["attribute"],
                    category_value=row["category_value"],
                    group_value=row["group_value"],
                    group_in_category=int(row["group_in_category"]),
                    group_outside_category=int(row["group_outside_category"]),
                    other_in_category=int(row["other_in_category"]),
                    other_outside_category=int(row["other_outside_category"]),
                    group_total=int(row["group_total"]),
                    other_total=int(row["other_total"]),
                    category_total=int(row["category_total"]),
                    non_category_total=int(row["non_category_total"]),
                    odds_ratio=float(row["odds_ratio"]),
                    p_value=float(row["p_value"]),
                    corrected_p_value=float(row["corrected_p_value"]),
                    reject_fdr=row["reject_fdr"].strip().lower() == "true",
                    statistic=None if row["statistic"].strip().lower() in {"", "nan"} else float(row["statistic"]),
                    test_name=row["test_name"],
                )
            )
    return rows


def _input_csv_path(input_dir: Path, model_alias: str, question_key: str, attribute: str) -> Path:
    return input_dir / model_alias / f"{question_key}_{attribute.lower()}__category_tests.csv"


def _combine_rows(rows: Iterable[DetailedRow]) -> list[DetailedRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.corrected_p_value,
            -_effect_strength(row.odds_ratio),
            -row.category_total,
            row.question_key,
            row.category_value,
            row.group_value,
        ),
    )


def _is_overrepresented(row: DetailedRow) -> bool:
    return row.group_value.strip().lower() != "unknown" and (row.odds_ratio > 1 or math.isinf(row.odds_ratio))


def _attr_name(attribute: str) -> str:
    return "region" if attribute == "Race" else "gender"


def _format_class_label(value: str, attribute: str) -> str:
    if attribute == "Race":
        normalized = value.strip()
        if normalized == "Central/Eastern Europe":
            return "CEE"
        if normalized == "United Kingdom / North America":
            return "UK/NA"
        if normalized in {"UK / NA", "UK/ NA", "UK /NA", "UK/NA"}:
            return "UK/NA"
        return normalized
    return value


def _format_category_label(value: str, attribute: str) -> str:
    if attribute == "Race":
        normalized = value.strip()
        if normalized == "Central/Eastern Europe":
            return "CEE"
        if normalized in {"UK / NA", "UK/ NA", "UK /NA", "UK/NA"}:
            return "UK/NA"
        return normalized
    return value


def _question_title(question_key: str) -> str:
    return question_key.upper()


def _effect_strength(odds_ratio: float) -> float:
    if odds_ratio == 0.0:
        return math.inf
    if math.isinf(odds_ratio):
        return math.inf
    return abs(math.log10(odds_ratio))


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _format_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "--"
    if math.isinf(value):
        return r"$\infty$"
    if value == 0:
        return f"{0:.{digits}f}"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _format_p(value: float) -> str:
    if value == 0:
        return r"$<10^{-300}$"
    if value < 1e-4:
        exponent = int(math.floor(math.log10(value)))
        return rf"$10^{{{exponent}}}$"
    return f"{value:.4f}"


def _significance_stars(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def _format_p_with_stars(value: float) -> str:
    stars = _significance_stars(value)
    if not stars:
        return _format_p(value)
    return rf"{_format_p(value)}\textsuperscript{{{stars}}}"


def _latex_table_header(caption: str, label: str, columns: str) -> list[str]:
    return [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"% //",
        r"\toprule",
    ]


def _latex_table_footer() -> list[str]:
    return [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pcol(width: str) -> str:
    return f"p{{{width}}}"


def _summary_columns() -> str:
    return "".join(
        [
            _pcol(_SUMMARY_WIDTHS["category"]),
            _pcol(_SUMMARY_WIDTHS["class"]),
            _pcol(_SUMMARY_WIDTHS["ncat"]),
            _pcol(_SUMMARY_WIDTHS["or"]),
            _pcol(_SUMMARY_WIDTHS["padj"]),
        ]
    )


def _appendix_columns() -> str:
    return "".join(
        [
            _pcol(_APPENDIX_WIDTHS["category"]),
            _pcol(_APPENDIX_WIDTHS["class"]),
            _pcol(_APPENDIX_WIDTHS["a"]),
            _pcol(_APPENDIX_WIDTHS["b"]),
            _pcol(_APPENDIX_WIDTHS["c"]),
            _pcol(_APPENDIX_WIDTHS["d"]),
            _pcol(_APPENDIX_WIDTHS["ngroup"]),
            _pcol(_APPENDIX_WIDTHS["nother"]),
            _pcol(_APPENDIX_WIDTHS["ncat"]),
            _pcol(_APPENDIX_WIDTHS["or"]),
            _pcol(_APPENDIX_WIDTHS["p"]),
            _pcol(_APPENDIX_WIDTHS["padj"]),
        ]
    )


def _appendix_header_row() -> str:
    return (
        r"\textbf{Category} & \textbf{class} & \textbf{group in category} & "
        r"\textbf{group out of category} & \textbf{other in category} & "
        r"\textbf{other out of category} & \textbf{$n_{group}$} & \textbf{$n_{other}$} & "
        rf"\textbf{{$n_{{cat}}$}} & \textbf{{OR}} & \textbf{{$p$}} & \textbf{{$p_{{adj}}$}} \\\\" 
    )


def _build_summary_table(rows: list[DetailedRow], question_key: str, attribute: str, output_path: Path) -> None:
    selected = _combine_rows([row for row in rows if _is_overrepresented(row)])[:8]
    attribute_label = _attr_name(attribute)
    question_label = _question_title(question_key)

    lines = _latex_table_header(
        caption=f"Top 8 {attribute_label} findings for {question_label} across all models.",
        label=f"tab:summary-{question_key}-{attribute_label.lower()}",
        columns=_summary_columns(),
    )
    lines.extend(
        [
            r"\textbf{Category} & \textbf{class} & \textbf{$n_{cat}$} & \textbf{OR} & \textbf{$p_{adj}$} \\",
            r"\midrule",
        ]
    )
    for row in selected:
        lines.append(
            " & ".join(
                [
                    _escape_latex(_format_category_label(row.category_value, attribute)),
                    _escape_latex(_format_class_label(row.group_value, attribute)),
                    str(row.category_total),
                    _format_float(row.odds_ratio, digits=3),
                    _format_p_with_stars(row.corrected_p_value),
                ]
            )
            + r" \\",
        )
        if row is not selected[-1]:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"{\footnotesize * $p_{adj} < 0.05$, ** $p_{adj} < 0.01$, *** $p_{adj} < 0.001$}",
            r"\end{table}",
        ]
    )
    _write_lines(output_path, lines)


def _build_appendix_table(rows: list[DetailedRow], title: str, label: str, output_path: Path) -> None:
    header_row = _appendix_header_row()
    lines = [
        r"\clearpage",
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{title}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\begin{{tabular}}{{{_appendix_columns()}}}",
        r"\toprule",
        header_row,
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _escape_latex(_format_category_label(row.category_value, row.attribute)),
                    _escape_latex(_format_class_label(row.group_value, row.attribute)),
                    str(row.group_in_category),
                    str(row.group_outside_category),
                    str(row.other_in_category),
                    str(row.other_outside_category),
                    str(row.group_total),
                    str(row.other_total),
                    str(row.category_total),
                    _format_float(row.odds_ratio, digits=3),
                    _format_p(row.p_value),
                    _format_p_with_stars(row.corrected_p_value),
                ]
            )
            + r" \\",
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{landscape}",
    ])
    _write_lines(output_path, lines)


def _collect_rows(input_dir: Path, model_alias: str, question_key: str, attribute: str) -> list[DetailedRow]:
    csv_path = _input_csv_path(input_dir, model_alias, question_key, attribute)
    if not csv_path.exists():
        return []
    rows = _read_rows(csv_path)
    return [row for row in rows if _is_overrepresented(row)]


def _appendix_output_paths(output_dir: Path, question_key: str, model_alias: str, attribute_name: str) -> list[Path]:
    appendix_dir = output_dir / "appendix"
    return [
        appendix_dir / f"{model_alias}_{attribute_name}.tex",
        appendix_dir / question_key / f"{model_alias}_{attribute_name}.tex",
    ]


def _significant_appendix_output_paths(output_dir: Path, question_key: str, model_alias: str, attribute_name: str) -> list[Path]:
    significant_dir = output_dir / "appendix_significant"
    return [
        significant_dir / f"{model_alias}_{attribute_name}.tex",
        significant_dir / question_key / f"{model_alias}_{attribute_name}.tex",
    ]


def build_latex_tables(input_dir: Path, output_dir: Path) -> list[Path]:
    outputs: list[Path] = []

    for question_key in ("q1", "q2"):
        for attribute in ("Gender", "Race"):
            rows = _collect_rows(input_dir, "all_models", question_key, attribute)
            attribute_name = _attr_name(attribute)
            summary_path = output_dir / f"summary_{question_key}_{attribute_name}.tex"
            _build_summary_table(rows, question_key, attribute, summary_path)
            outputs.append(summary_path)

    for question_key in ("q1", "q2"):
        for model_alias in _MODEL_ORDER:
            for attribute in ("Gender", "Race"):
                rows = _collect_rows(input_dir, model_alias, question_key, attribute)
                if not rows:
                    continue
                attribute_name = _attr_name(attribute)
                sorted_rows = sorted(
                    rows,
                    key=lambda row: (row.corrected_p_value, -_effect_strength(row.odds_ratio), -row.category_total, row.category_value, row.group_value),
                )
                for appendix_path in _appendix_output_paths(output_dir, question_key, model_alias, attribute_name):
                    _build_appendix_table(
                        rows=sorted_rows,
                        title=f"{model_alias} {attribute_name} significance results for {question_key.upper()}.",
                        label=f"tab:appendix-{question_key}-{model_alias}-{attribute_name}",
                        output_path=appendix_path,
                    )
                    outputs.append(appendix_path)

                significant_rows = [row for row in sorted_rows if row.reject_fdr]
                if model_alias == "all_models" or significant_rows:
                    for appendix_path in _significant_appendix_output_paths(output_dir, question_key, model_alias, attribute_name):
                        _build_appendix_table(
                            rows=significant_rows,
                            title=f"{model_alias} {attribute_name} significant results for {question_key.upper()}",
                            label=f"tab:appendix-significant-{question_key}-{model_alias}-{attribute_name}",
                            output_path=appendix_path,
                        )
                        outputs.append(appendix_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LaTeX summary and appendix tables from significance CSVs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help="Directory containing the significance table CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where LaTeX fragments will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_latex_tables(args.input_dir, args.output_dir)
    print("Wrote LaTeX tables:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()