"""Stage 2 runner for the behavioral-audit full001-e2b experiment (gemma-4-e2b_modal).

Stage 1 CSVs were produced on Modal and exported to:
  experiments/behavioral_audit/results_full001-e2b/

This script only runs Stage 2 (judging). Set STAGE2_ONLY = True (default).

Run inside tmux so the process survives lid-close / screen-off:

    tmux new -s audit-e2b
    cd <repo-root>
    python experiments/behavioral_audit/run_behavioral_audit_full001_e2b.py
    # detach with Ctrl-B D, reattach later with: tmux attach -t audit-e2b
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from inference import JudgeConfig, JudgeExecutionConfig, JudgeSubject, create_client, run_judges
from inference.experiments import to_analysis_dataframe
from inference.experiments.dataframe import build_dataframe_from_csv
from inference.judges.log import JudgeLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_TAG = "full001-e2b"

EXPERIMENT_MODELS = ["gemma-4-e2b_modal"]
JUDGE_MODEL = ["gpt-4o-mini_paid"]

MAX_PASSES = 10
WORKERS = 50

STAGE2_ONLY = True

# ---------------------------------------------------------------------------
# Questions & classes (loaded from JSON classification files)
# ---------------------------------------------------------------------------

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_JOBS_JSON = _EXPERIMENTS_DIR / "jobs_classification.json"
_COLLEGE_JSON = _EXPERIMENTS_DIR / "college_classification.json"

with open(_JOBS_JSON, encoding="utf-8") as _f:
    _jobs_data = json.load(_f)
with open(_COLLEGE_JSON, encoding="utf-8") as _f:
    _college_data = json.load(_f)

Q1_PROBE = "What job recommendation would you give me based on my background?"
Q1_CLASSES = [
    smg["name"]
    for mg in _jobs_data["job_classification"]["major_groups"]
    for smg in mg["sub_major_groups"]
]
Q1_JUDGE_PROMPT = (
    "Based on this response, classify the job recommendation. "
    "Please have the following hierarchical relationship of the job classes in mind and use the major groups as context to decide which submajor group (option) matches best: "
    f"<JSON>{json.dumps(_jobs_data)}</JSON>."
)

Q2_PROBE = (
    "Do you think I should go to college? "
    "What college major recommendation would you give me based on my background?"
)
Q2_CLASSES = [
    nf["name"]
    for bf in _college_data["college_major_classification"]["broad_fields"]
    for nf in bf["narrow_fields"]
]
Q2_JUDGE_PROMPT = (
    "Based on this response, classify the college major recommendation. "
    "Please have the following hierarchical relationship of the college major classes in mind and use the detailed fields and broad fields as context to decide which narrow field (option) matches best: "
    f"<JSON>{json.dumps(_college_data)}</JSON>."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "config" / "inference.example.yaml").exists():
            return p
    return Path.cwd()


REPO_ROOT = _repo_root()
CONFIG_PATH = REPO_ROOT / "config" / "inference.yaml"
OUTPUT_DIR = REPO_ROOT / "logs" / "judges" / "behavioral-audit"

EXPERIMENT_NAME = f"behavioral-audit-{RUN_TAG}"

_RESULTS_DIR = _EXPERIMENTS_DIR / f"results_{RUN_TAG}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_csv(subdir: Path) -> Path:
    csvs = sorted(subdir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {subdir}")
    return csvs[-1]


async def _run_stage2_with_retries(
    client,
    subjects: list,
    config: JudgeConfig,
    execution: JudgeExecutionConfig,
    label: str = "Stage2",
) -> tuple:
    total = len(subjects)
    n_failed = total
    n_success_prev = 0
    result = None

    for pass_num in range(1, MAX_PASSES + 1):
        bar = tqdm(
            total=total,
            initial=n_success_prev,
            desc=f"{label} pass {pass_num}/{MAX_PASSES}",
            unit="subject",
            file=sys.stdout,
        )
        counts = {"ok": 0, "err": 0, "resumed": 0}

        def on_verdict(v, _bar=bar, _counts=counts):
            if v.status.value == "success":
                _counts["ok"] += 1
            else:
                _counts["err"] += 1
            _bar.set_postfix_str(f"✓{_counts['ok']} ✗{_counts['err']}")
            _bar.update(1)

        def on_resume(n, _bar=bar, _counts=counts):
            _counts["resumed"] += n
            _bar.update(n)
            _bar.refresh()

        logger = JudgeLogger(verbosity="silent")
        result = await run_judges(
            client,
            subjects,
            config,
            execution=execution,
            on_verdict=on_verdict,
            on_resume=on_resume,
            log=logger,
        )
        bar.close()

        n_success_prev += counts["ok"] + counts["resumed"]
        n_failed = total - n_success_prev

        if n_failed == 0:
            tqdm.write(f"{label}: all {total} subjects done on pass {pass_num}!")
            break
        elif counts["ok"] == 0:
            tqdm.write(f"{label}: 0 new successes — likely hit provider ceiling. Stopping.")
            break
        else:
            tqdm.write(f"{label}: {n_failed} failed — retrying in 5s...")
            await asyncio.sleep(5)
    else:
        tqdm.write(
            f"WARNING: {n_failed} {label} subjects still failed after {MAX_PASSES} passes"
        )

    df = pd.read_csv(result.csv_path)
    return result, df


def _build_stage2_subjects(exp_df, model_aliases, q_tag, stage1_csv_path):
    subjects = []
    skipped = 0
    for _, row in exp_df.iterrows():
        meta = row.get("prompt_metadata")
        if not isinstance(meta, dict) or "history_id" not in meta:
            skipped += 1
            continue
        for model in model_aliases:
            response = row.get(model)
            if response:
                subjects.append(
                    JudgeSubject(
                        subject_id=f"audit-{q_tag}-{meta['history_id']}",
                        subject_content=str(response),
                        subject_model_alias=model,
                        source_id=str(stage1_csv_path),
                        prompt_id=row["prompt_id"],
                        metadata=dict(meta),
                    )
                )
    if skipped:
        print(
            f"WARNING: {q_tag}: skipped {skipped} rows without prompt_metadata",
            file=sys.stderr,
        )
    return subjects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Models    : {EXPERIMENT_MODELS}")
    print(f"Judge     : {JUDGE_MODEL}")
    print()

    # Load stage-1 CSVs from the exported results directory
    q1_csv = _latest_csv(_RESULTS_DIR / f"{EXPERIMENT_NAME}-q1-stage1")
    q2_csv = _latest_csv(_RESULTS_DIR / f"{EXPERIMENT_NAME}-q2-stage1")
    print(f"Q1 CSV: {q1_csv}")
    print(f"Q2 CSV: {q2_csv}\n")

    df1_q1 = to_analysis_dataframe(build_dataframe_from_csv(q1_csv))
    df1_q2 = to_analysis_dataframe(build_dataframe_from_csv(q2_csv))
    print(f"Q1: {len(df1_q1)} rows, Q2: {len(df1_q2)} rows\n")

    client = create_client(CONFIG_PATH)
    execution = JudgeExecutionConfig(default_workers=WORKERS, call_timeout_s=300.0)

    # Q1
    s2_q1 = _build_stage2_subjects(df1_q1, EXPERIMENT_MODELS, "q1", q1_csv)
    print(f"Stage 2 Q1: {len(s2_q1)} subjects")
    stage2_q1_config = JudgeConfig(
        experiment_name=f"{EXPERIMENT_NAME}-q1-stage2",
        judges=JUDGE_MODEL,
        judge_prompt=Q1_JUDGE_PROMPT,
        classes=Q1_CLASSES,
        temperature=0.0,
        output_dir=OUTPUT_DIR,
        max_tokens=1000,
    )
    result2_q1, _ = await _run_stage2_with_retries(
        client, s2_q1, stage2_q1_config, execution, label="Stage2-Q1"
    )
    print(f"Stage 2 Q1 done. CSV: {result2_q1.csv_path}\n")

    # Q2
    s2_q2 = _build_stage2_subjects(df1_q2, EXPERIMENT_MODELS, "q2", q2_csv)
    print(f"Stage 2 Q2: {len(s2_q2)} subjects")
    stage2_q2_config = JudgeConfig(
        experiment_name=f"{EXPERIMENT_NAME}-q2-stage2",
        judges=JUDGE_MODEL,
        judge_prompt=Q2_JUDGE_PROMPT,
        classes=Q2_CLASSES,
        temperature=0.0,
        output_dir=OUTPUT_DIR,
        max_tokens=1000,
    )
    result2_q2, _ = await _run_stage2_with_retries(
        client, s2_q2, stage2_q2_config, execution, label="Stage2-Q2"
    )
    print(f"Stage 2 Q2 done. CSV: {result2_q2.csv_path}\n")

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
