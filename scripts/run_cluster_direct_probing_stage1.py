#!/usr/bin/env python3
"""Cluster direct-probing Stage 1 via local vLLM. See scripts/README.md."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from inference.config import load_config_from_file
from inference.experiments.vllm_matrix import MatrixColumnRun, run_matrix_column

_EXIT_MESSAGES = {
    2: "config or model alias error",
    3: "vLLM server not reachable or wrong served model",
    4: "no prompts to run",
    5: "connection circuit breaker tripped (server outage)",
    6: "CSV prompt set mismatch — resuming would delete rows",
    10: "partial column — some cells not SUCCESS",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster direct-probing Stage 1 (vLLM). Notebooks: run_direct_probing.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to inference YAML (vLLM config).")
    parser.add_argument(
        "--model-alias",
        required=True,
        help="Alias KEY of the column to fill (YAML dict key / notebook EXPERIMENT_MODEL).",
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Comma-separated full matrix column set. Defaults to --model-alias only.",
    )
    parser.add_argument(
        "--experiment-name",
        default="",
        help="Output label. Defaults to vllm-<model-alias>.",
    )
    parser.add_argument(
        "--csv-path",
        default="",
        help="Stable matrix CSV. Defaults to logs/<experiment-name>/matrix.csv.",
    )
    parser.add_argument(
        "--prompts-source",
        choices=["direct-probing", "demo"],
        default="direct-probing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N prompts (N=1 smoke test).",
    )
    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=10_000,
        help="direct-probing: personas per (Gender, Region) group.",
    )
    parser.add_argument(
        "--circuit-breaker-threshold",
        type=int,
        default=5,
        help="Consecutive connection errors before aborting fast.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip pre-run server probe (launcher already verified readiness).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")

    args = build_arg_parser().parse_args(argv)
    model_alias = args.model_alias
    columns = [c.strip() for c in args.columns.split(",") if c.strip()] or [model_alias]
    experiment_name = args.experiment_name or f"vllm-{model_alias}"
    csv_path = (
        Path(args.csv_path) if args.csv_path else Path("logs") / experiment_name / "matrix.csv"
    )
    config_path = Path(args.config)

    if args.limit is not None and args.limit < 0:
        print("ERROR: --limit must be >= 0", file=sys.stderr)
        return 2

    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    config = load_config_from_file(config_path)
    if model_alias not in config.model_aliases:
        print(
            f"ERROR: --model-alias {model_alias!r} is not in {config_path}. "
            f"Known aliases: {sorted(config.model_aliases)}",
            file=sys.stderr,
        )
        return 2

    run = MatrixColumnRun(
        config_path=config_path,
        model_alias=model_alias,
        columns=columns,
        experiment_name=experiment_name,
        csv_path=csv_path,
        prompts_source=args.prompts_source,
        limit=args.limit,
        sample_per_group=args.sample_per_group,
        circuit_breaker_threshold=args.circuit_breaker_threshold,
        skip_probe=args.skip_probe,
    )

    exit_code = asyncio.run(run_matrix_column(run))
    if exit_code != 0 and exit_code in _EXIT_MESSAGES:
        print(f"ERROR: {_EXIT_MESSAGES[exit_code]}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
