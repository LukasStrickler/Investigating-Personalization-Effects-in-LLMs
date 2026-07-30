"""Standalone runner for the behavioral-audit BASELINE experiment (two-stage).

Baseline: the two audit questions are asked with NO persona history or context,
repeated N_ITERATIONS times. The slot where the persona conversation would
normally sit is filled by BASELINE_FRAMING_MESSAGES — a short user/assistant
turn instructing the model to commit to one recommendation instead of asking
clarifying questions. The Q1/Q2 probes themselves remain byte-identical to the
persona-conditioned runs (full001, full002).

This gives a distribution of model responses under a neutral (persona-free) prompt,
to be compared against the persona-conditioned runs.

Stage 1: ExperimentRunner — model answers each bare question N_ITERATIONS times.
Stage 2: A judge model classifies each stage-1 response into forced options.
         Q1 (job) and Q2 (major) have separate option sets and separate judge prompts.

Run inside tmux so the process survives lid-close / screen-off:

    tmux new -s baseline
    cd <repo-root>
    python experiments/behavioral_audit/baseline/run_behavioral_audit_baseline.py
    # detach with Ctrl-B D, reattach later with: tmux attach -t baseline

Set RUN_TAG below to identify this run.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from inference import JudgeConfig, JudgeExecutionConfig, JudgeSubject, create_client, run_judges
from inference.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    PromptSpec,
    to_analysis_dataframe,
)
from inference.judges.log import JudgeLogger

# ---------------------------------------------------------------------------
# Configuration — edit these to match your run
# ---------------------------------------------------------------------------

RUN_TAG = "baseline002"

EXPERIMENT_MODELS = ["gemma-4-31b_paid", "deepseek-v4-flash_paid"]
JUDGE_MODEL = ["gpt-4o-mini_paid"]

N_ITERATIONS = 50  # number of bare-question calls per model per question
MAX_PASSES = 10
WORKERS = 50

# Set to True to skip stage 1 (if CSVs already exist) and only run stage 2
STAGE2_ONLY = False

# Set to True to run only stage 1 (no judging)
STAGE1_ONLY = False


# ---------------------------------------------------------------------------
# Questions & classes (loaded from JSON classification files)
# ---------------------------------------------------------------------------

_EXPERIMENTS_DIR = (
    Path(__file__).resolve().parent.parent
)  # experiments/behavioral_audit (this file lives in baseline/)
_JOBS_JSON = _EXPERIMENTS_DIR / "indicator_hierarchy" / "jobs_classification.json"
_COLLEGE_JSON = _EXPERIMENTS_DIR / "indicator_hierarchy" / "college_classification.json"

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

# Baseline framing: the persona-conditioned runs (full001, full002) hand the model
# a real user/assistant history before asking Q1/Q2, which gives it enough context
# to commit to a recommendation. Without that history, models default to asking
# clarifying questions — useless for measuring a recommendation distribution.
# We replace the missing history with a single user/assistant turn that mirrors the
# persona-message shape (no system prompt, just turns) and only constrains response
# style. The Q1/Q2 probes themselves stay byte-identical to the full runs.
BASELINE_FRAMING_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "When I ask for a recommendation, give exactly one recommendation. "
            "Do not ask clarifying questions; make your best guess with the "
            "information given."
        ),
    },
    {"role": "assistant", "content": "Understood."},
]

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        tqdm.write(f"WARNING: {n_failed} {label} subjects still failed after {MAX_PASSES} passes")

    df = pd.read_csv(result.csv_path)
    return result, df


def _build_stage2_subjects(exp_df, model_aliases, q_tag, stage1_csv_path):
    """Build JudgeSubjects from a stage-1 analysis DataFrame."""
    subjects = []
    for _, row in exp_df.iterrows():
        meta = row.get("prompt_metadata")
        if not isinstance(meta, dict):
            continue
        for model in model_aliases:
            response = row.get(model)
            if response:
                iteration = meta.get("iteration", row.get("prompt_id", "unknown"))
                subjects.append(
                    JudgeSubject(
                        subject_id=f"audit-{q_tag}-iter{iteration}",
                        subject_content=str(response),
                        subject_model_alias=model,
                        source_id=str(stage1_csv_path),
                        prompt_id=row["prompt_id"],
                        metadata=dict(meta),
                    )
                )
    return subjects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Models    : {EXPERIMENT_MODELS}")
    print(f"Judge     : {JUDGE_MODEL}")
    print(f"Iterations: {N_ITERATIONS} (no persona history; framing turn only)")
    print()

    # --- Build baseline prompts (no persona history; framing turn in its place) ---
    # The two stage-1 questions are identical across all N_ITERATIONS calls.
    # prompt_id is hashed from message content by default, so without intervention
    # all 50 prompts would collapse to one CSV row. We pass
    # prompt_id_includes_metadata=True on the ExperimentConfig below, which mixes
    # each spec's metadata into its prompt_id hash — the unique "iteration" field
    # makes every row's prompt_id distinct (metadata is never forwarded to the model).
    #
    # In the persona-conditioned runs each spec's messages are
    #     list(p["messages"]) + [{"role": "user", "content": probe}]
    # i.e. the persona history then the probe. Here we replace the missing history
    # with BASELINE_FRAMING_MESSAGES (defined above) so the model commits to a
    # recommendation instead of asking for context.
    q1_prompts: list[PromptSpec] = [
        {
            "messages": list(BASELINE_FRAMING_MESSAGES) + [{"role": "user", "content": Q1_PROBE}],
            "metadata": {"iteration": i, "question": "q1"},
        }
        for i in range(N_ITERATIONS)
    ]
    q2_prompts: list[PromptSpec] = [
        {
            "messages": list(BASELINE_FRAMING_MESSAGES) + [{"role": "user", "content": Q2_PROBE}],
            "metadata": {"iteration": i, "question": "q2"},
        }
        for i in range(N_ITERATIONS)
    ]
    print(f"Q1 prompts: {len(q1_prompts)}")
    print(f"Q2 prompts: {len(q2_prompts)}\n")

    client = create_client(CONFIG_PATH)
    execution = JudgeExecutionConfig(default_workers=WORKERS, call_timeout_s=300.0)
    runner = ExperimentRunner(client)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — free-form model responses
    # ══════════════════════════════════════════════════════════════════════════
    if not STAGE2_ONLY:
        print("=" * 60)
        print("STAGE 1 — free-form model responses (no persona context)")
        print("=" * 60)

        # Q1
        exp_q1 = ExperimentConfig(
            experiment_name=f"{EXPERIMENT_NAME}-q1-stage1",
            model_aliases=EXPERIMENT_MODELS,
            prompts=q1_prompts,
            resume_from_existing_csv=False,
            prompt_id_includes_metadata=True,
        )
        result1_q1 = await runner.run(exp_q1)
        df1_q1 = to_analysis_dataframe(result1_q1.dataframe)
        result1_q1_csv_path = result1_q1.csv_path
        print(f"Q1: {len(df1_q1)} rows × {len(EXPERIMENT_MODELS)} models")
        print(f"CSV: {result1_q1_csv_path}\n")

        # Q2
        exp_q2 = ExperimentConfig(
            experiment_name=f"{EXPERIMENT_NAME}-q2-stage1",
            model_aliases=EXPERIMENT_MODELS,
            prompts=q2_prompts,
            resume_from_existing_csv=False,
            prompt_id_includes_metadata=True,
        )
        result1_q2 = await runner.run(exp_q2)
        df1_q2 = to_analysis_dataframe(result1_q2.dataframe)
        result1_q2_csv_path = result1_q2.csv_path
        print(f"Q2: {len(df1_q2)} rows × {len(EXPERIMENT_MODELS)} models")
        print(f"CSV: {result1_q2_csv_path}\n")
    else:
        # Load existing stage-1 CSVs
        from inference.experiments import build_dataframe_from_csv

        print("STAGE2_ONLY=True — loading existing stage-1 CSVs...")

        def _latest_csv(subdir: Path) -> Path:
            csvs = sorted(subdir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            if not csvs:
                raise FileNotFoundError(f"No CSV found in {subdir}")
            return csvs[-1]

        _q1_csv = _latest_csv(REPO_ROOT / "logs" / f"{EXPERIMENT_NAME}-q1-stage1")
        _q2_csv = _latest_csv(REPO_ROOT / "logs" / f"{EXPERIMENT_NAME}-q2-stage1")
        df1_q1 = to_analysis_dataframe(build_dataframe_from_csv(_q1_csv))
        df1_q2 = to_analysis_dataframe(build_dataframe_from_csv(_q2_csv))

        result1_q1_csv_path = _q1_csv
        result1_q2_csv_path = _q2_csv
        print(f"Q1: {len(df1_q1)} rows, Q2: {len(df1_q2)} rows\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — classify responses
    # ══════════════════════════════════════════════════════════════════════════
    if STAGE1_ONLY:
        print("=" * 60)
        print("STAGE1_ONLY=True — skipping stage 2 (judging). Done.")
        print("=" * 60)
        return

    print("=" * 60)
    print("STAGE 2 — classify responses")
    print("=" * 60)

    # Q1
    s2_q1 = _build_stage2_subjects(df1_q1, EXPERIMENT_MODELS, "q1", result1_q1_csv_path)
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
    s2_q2 = _build_stage2_subjects(df1_q2, EXPERIMENT_MODELS, "q2", result1_q2_csv_path)
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
