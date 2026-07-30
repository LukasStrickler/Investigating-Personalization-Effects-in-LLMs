#!/usr/bin/env python3
"""Cluster direct-probing Stage 2 judge (OpenRouter). See scripts/README.md."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from inference import JudgeConfig, JudgeExecutionConfig, create_client, run_judges
from inference.config import load_config_from_file, resolve_api_key
from inference.experiments import to_analysis_dataframe
from inference.experiments.dataframe import build_dataframe_from_csv
from inference.experiments.persistence import load_existing_matrix
from inference.experiments.vllm_matrix import (
    DIRECT_PROBING_JUDGE_PROMPT,
    build_direct_probing_judge_subjects,
    load_direct_probing_stage1,
)

_EXIT_MESSAGES = {
    2: "config, CSV, personas, or judge setup error",
    4: "no judge subjects (empty Stage-1 column?)",
    10: "partial judge run — some verdicts failed",
}


def _expected_verdict_count(*, n_subjects: int, n_judges: int) -> int:
    return n_subjects * n_judges


def _validate_stage2_config(config_path: Path, judge_aliases: list[str]) -> str | None:
    """Return an error message, or None if config is usable for Stage 2."""
    config = load_config_from_file(config_path)
    if config.default_provider == "vllm":
        return (
            "config still targets vllm (offline GPU). Stage 2 needs OpenRouter: "
            "cp config/inference.example.yaml config/inference.yaml"
        )
    for alias in judge_aliases:
        alias_cfg = config.model_aliases.get(alias)
        if alias_cfg is None:
            return f"judge alias {alias!r} not in {config_path}"
        provider = config.providers.get(alias_cfg.provider)
        if provider is None:
            return f"provider {alias_cfg.provider!r} missing from config"
        if provider.name == "vllm":
            return f"judge alias {alias!r} uses vllm; Stage 2 needs an online provider (OpenRouter)"
        try:
            resolve_api_key(provider)
        except ValueError as error:
            return str(error)
    return None


async def run_stage2(
    *,
    config_path: Path,
    csv_path: Path,
    model_alias: str,
    judge_aliases: list[str],
    experiment_name: str,
    output_dir: Path,
    workers: int,
    sample_per_group: int,
) -> int:
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 2
    if not csv_path.exists():
        print(f"ERROR: Stage-1 CSV not found: {csv_path}", file=sys.stderr)
        return 2

    if message := _validate_stage2_config(config_path, judge_aliases):
        print(f"ERROR: {message}", file=sys.stderr)
        return 2

    try:
        expected_prompts, combined_classes = load_direct_probing_stage1(
            sample_per_group=sample_per_group
        )
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    raw_df = build_dataframe_from_csv(csv_path)
    df = to_analysis_dataframe(raw_df)

    if len(raw_df) != len(expected_prompts):
        print(
            f"WARNING: CSV has {len(raw_df)} rows but sample_per_group={sample_per_group} "
            f"implies {len(expected_prompts)} — pass the same --sample-per-group as Stage 1",
            file=sys.stderr,
        )

    marker = Path(f"{csv_path}.{model_alias}.complete")
    if not marker.exists():
        print(
            f"WARNING: Stage 1 marker missing ({marker}); column may be incomplete",
            file=sys.stderr,
        )

    if model_alias not in df.columns:
        model_cols = [c for c in df.columns if c not in ("prompt_id", "prompt", "prompt_metadata")]
        print(
            f"ERROR: column {model_alias!r} not in {csv_path}. Columns: {model_cols}",
            file=sys.stderr,
        )
        return 2

    _seen, completed = load_existing_matrix(csv_path)
    success_count = sum(1 for (_pid, alias) in completed if alias == model_alias)
    print(f"[stage2] csv={csv_path} column={model_alias} success_cells={success_count}")

    subjects, skipped = build_direct_probing_judge_subjects(
        df, model_alias=model_alias, csv_path=csv_path
    )
    if skipped:
        print(f"WARNING: skipped {skipped} rows without prompt_metadata", file=sys.stderr)
    if not subjects:
        print("ERROR: no judge subjects (empty column or no SUCCESS cells?)", file=sys.stderr)
        return 4

    print(f"[stage2] {len(subjects)} subjects → judges {judge_aliases}")

    client = create_client(config_path)
    judge_config = JudgeConfig(
        experiment_name=experiment_name,
        judges=judge_aliases,
        judge_prompt=DIRECT_PROBING_JUDGE_PROMPT,
        classes=combined_classes,
        temperature=0.0,
        output_dir=output_dir,
    )
    execution = JudgeExecutionConfig(default_workers=workers)
    result = await run_judges(client, subjects, judge_config, execution=execution)
    expected = _expected_verdict_count(n_subjects=len(subjects), n_judges=len(judge_aliases))
    n_ok = sum(1 for v in result.verdicts if v.status.value == "success")
    print(f"[stage2] done: {result.csv_path} ({n_ok}/{expected} verdicts success)")
    return 0 if n_ok == expected else 10


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster direct-probing Stage 2 (OpenRouter judge on Stage-1 CSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/inference.yaml",
        help="Judge provider config (typically OpenRouter after cp inference.example.yaml).",
    )
    parser.add_argument("--csv-path", required=True, help="Stage-1 matrix CSV path.")
    parser.add_argument(
        "--model-alias",
        required=True,
        help="Stage-1 column alias (e.g. gemma-4-31b).",
    )
    parser.add_argument(
        "--judge-alias",
        action="append",
        dest="judge_aliases",
        required=True,
        help="Judge model alias from config (repeat for multiple judges).",
    )
    parser.add_argument(
        "--experiment-name",
        default="",
        help="Judgment CSV label. Defaults to <csv-stem>-stage2.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/judges/direct-probing",
        help="Judgment CSV output directory.",
    )
    parser.add_argument("--workers", type=int, default=5, help="Concurrent judge workers.")
    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=10_000,
        help="Must match Stage-1 sampling (defines judge class labels).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    csv_path = Path(args.csv_path)
    experiment_name = args.experiment_name or f"{csv_path.stem}-stage2"
    return asyncio.run(
        run_stage2(
            config_path=Path(args.config),
            csv_path=csv_path,
            model_alias=args.model_alias,
            judge_aliases=list(args.judge_aliases),
            experiment_name=experiment_name,
            output_dir=Path(args.output_dir),
            workers=args.workers,
            sample_per_group=args.sample_per_group,
        )
    )


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0 and exit_code in _EXIT_MESSAGES:
        print(f"ERROR: {_EXIT_MESSAGES[exit_code]}", file=sys.stderr)
    sys.exit(exit_code)
