"""Stage-1 + Stage-2 baseline runner for a Modal-hosted subject model.

This is the Modal sibling of ``run_behavioral_audit_baseline.py``: the same
persona-free control (framing turn + Q1/Q2 probes, ``N_ITERATIONS`` repeats),
but the subject model is a vLLM server on Modal and run parameters are passed
on the command line instead of editing module constants.

Unlike the personalized Modal path (``run_behavioral_audit_modal.py``), this
runner executes **Stage 1 and Stage 2 in one process** — the OpenRouter judge
alias (``gpt-4o-mini_paid``) is already in ``config/inference.modal.example.yaml``.

Prerequisites
    modal deploy experiments/modal_gpu_poc/modal_serve.py
    export MODAL_BASE_URL="https://<workspace>--pers-subject-serve-serve.modal.run/v1"
    export MODAL_API_KEY="EMPTY"
    cp config/inference.modal.example.yaml config/inference.yaml

Usage
    python experiments/behavioral_audit/run_behavioral_audit_baseline_modal.py \
        --run-tag baseline-e2b --subject-alias gemma-4-e2b_modal

    python experiments/behavioral_audit/run_behavioral_audit_baseline_modal.py \
        --run-tag baseline-ministral3-8b --subject-alias ministral-3-8b_modal

Export committable results with ``export_results.py`` once the run finishes.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import run_behavioral_audit_baseline as baseline


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-tag", default="baseline-e2b",
                    help="names the logs/ and results_ output dirs (behavioral-audit-<tag>)")
    ap.add_argument("--subject-alias", required=True, metavar="ALIAS",
                    help="stage-1 model alias from config/inference.yaml (exactly one Modal subject)")
    ap.add_argument("--judge", action="append", default=None, metavar="ALIAS",
                    help="stage-2 judge alias (repeatable). Default: gpt-4o-mini_paid")
    ap.add_argument("--config", type=Path, default=baseline.CONFIG_PATH,
                    help="inference config (use config/inference.modal.example.yaml for Modal subjects)")
    ap.add_argument("--n-iterations", type=int, default=baseline.N_ITERATIONS,
                    help="bare-question calls per model per question")
    ap.add_argument("--stage1-only", action="store_true", help="generate responses, skip judging")
    ap.add_argument("--stage2-only", action="store_true", help="judge existing stage-1 CSVs only")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config.exists():
        raise SystemExit(
            f"config not found: {args.config}\n"
            "cp config/inference.modal.example.yaml config/inference.yaml"
        )

    baseline.RUN_TAG = args.run_tag
    baseline.EXPERIMENT_MODELS = [args.subject_alias]
    if args.judge:
        baseline.JUDGE_MODEL = args.judge
    baseline.CONFIG_PATH = args.config
    baseline.N_ITERATIONS = args.n_iterations
    baseline.STAGE1_ONLY = args.stage1_only
    baseline.STAGE2_ONLY = args.stage2_only
    baseline.EXPERIMENT_NAME = f"behavioral-audit-{args.run_tag}"
    asyncio.run(baseline.main())


if __name__ == "__main__":
    main()
