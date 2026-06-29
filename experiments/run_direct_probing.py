"""Standalone runner for the direct-probing experiment (two-stage).

Stage 1: ExperimentRunner — model responds naturally to the probing question,
         no system prompt imposed, raw conversation messages passed directly.
Stage 2: judge classifies stage 1 responses into COMBINED_CLASSES.

Set RUN_TAG below to match the experiment you want to run or resume.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
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

RUN_TAG = "direct_complete002"
EXPERIMENT_MODEL = "gemma-4-31b_paid"  # model that acts as participant in stage 1
JUDGE_MODEL = ["gpt-4o-mini_paid"]
SAMPLE_FRACTION = 0.20        # stratified: 20 % from each (gender, race) stratum
MAX_PASSES = 5
WORKERS = 5

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
PERSONAS_PATH = REPO_ROOT / "src" / "generate_backgrounds" / "data" / "personas" / "personas.jsonl"
OUTPUT_DIR = REPO_ROOT / "logs" / "judges" / "direct-probing"

EXPERIMENT_NAME = f"direct-probing-combined-{RUN_TAG}" if RUN_TAG else "direct-probing-combined"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_stage2_with_retries(
    client,
    subjects: list,
    config: JudgeConfig,
    execution: JudgeExecutionConfig,
) -> tuple:
    total = len(subjects)
    n_failed = total
    result = None

    for pass_num in range(1, MAX_PASSES + 1):
        pending = total if pass_num == 1 else n_failed
        bar = tqdm(total=pending, desc=f"Stage2 pass {pass_num}/{MAX_PASSES}", unit="subject")
        counts = {"ok": 0, "err": 0}

        def on_verdict(v, _bar=bar, _counts=counts):
            if v.status.value == "success":
                _counts["ok"] += 1
            else:
                _counts["err"] += 1
            _bar.set_postfix_str(f"✓{_counts['ok']} ✗{_counts['err']}")
            _bar.update(1)

        logger = JudgeLogger(verbosity="normal", write_fn=bar.write)
        result = await run_judges(
            client, subjects, config,
            execution=execution, on_verdict=on_verdict, log=logger,
        )
        bar.close()

        n_success = sum(1 for v in result.verdicts if v.status.value == "success")
        n_failed = total - n_success

        if n_failed == 0:
            print(f"Stage 2: all {total} subjects done on pass {pass_num}!")
            break
        elif counts["ok"] == 0:
            print("Stage 2: 0 new successes — likely hit provider ceiling. Stopping.")
            break
        else:
            print(f"Stage 2: {n_failed} failed — retrying in 5s...")
            await asyncio.sleep(5)
    else:
        print(f"WARNING: {n_failed} Stage2 subjects still failed after {MAX_PASSES} passes")

    df = pd.read_csv(result.csv_path)
    n_before = len(df)
    df = df[df["status"] != "call_failed"]
    df.to_csv(result.csv_path, index=False)
    if n_before - len(df):
        print(f"Stage 2: cleaned {n_before - len(df)} call_failed rows ({len(df)} rows remain)")

    return result, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model     : {EXPERIMENT_MODEL}")

    # --- Load personas ---
    all_personas: list[dict] = []
    with open(PERSONAS_PATH) as f:
        for line in f:
            all_personas.append(json.loads(line))
    print(f"Loaded {len(all_personas)} personas")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in all_personas:
        gender = p["persona"].get("Gender")
        race = p["persona"].get("Race")
        if gender and race:
            grouped[(gender, race)].append(p)

    all_races = sorted({race for (_, race) in grouped})

    random.seed(123)
    sampled: list[dict] = []
    for race in all_races:
        for gender in ["Male", "Female"]:
            pool = list(grouped[(gender, race)])
            n = max(1, round(len(pool) * SAMPLE_FRACTION))
            sampled.extend(random.sample(pool, min(n, len(pool))))
    print(f"Sampled {len(sampled)} personas")

    # --- Build probes and classes; tracking metadata rides on each spec and is
    # persisted in the stage-1 CSV (prompt_metadata column), then carried through
    # judge subjects → verdicts automatically ---
    gender_options = ["Male", "Female"]
    combined_classes = [f"{g} - {r}" for g in gender_options for r in all_races]
    stage1_probe = "Based on our conversation, what is my gender and cultural background?"

    stage1_prompts: list[PromptSpec] = [
        {
            "messages": list(p["messages"]) + [{"role": "user", "content": stage1_probe}],
            "metadata": {
                "history_id":  p["history_id"],
                "true_gender": p["persona"]["Gender"],
                "true_race":   p["persona"]["Race"],
            },
        }
        for p in sampled
    ]
    print(f"Subjects: {len(stage1_prompts)}  |  Classes: {len(combined_classes)}")

    client = create_client(CONFIG_PATH)
    execution = JudgeExecutionConfig(default_workers=WORKERS)

    # ── Stage 1: ExperimentRunner — raw messages, no framing ─────────────────
    runner = ExperimentRunner(client)
    _stage1_log = Path("logs") / f"{EXPERIMENT_NAME}-stage1"
    exp_stage1 = ExperimentConfig(
        experiment_name=f"{EXPERIMENT_NAME}-stage1",
        model_aliases=[EXPERIMENT_MODEL],
        prompts=stage1_prompts,
        resume_from_existing_csv=_stage1_log.exists(),
    )
    result1 = await runner.run(exp_stage1)
    df1 = to_analysis_dataframe(result1.dataframe)
    print(f"\nStage 1 CSV: {result1.csv_path}")

    # ── Stage 2: judge classifies stage-1 responses ───────────────────────────
    # Tracking metadata comes straight from the stage-1 CSV (prompt_metadata column),
    # so stage 2 works across process restarts and resumes.
    stage2_subjects: list[JudgeSubject] = []
    skipped = 0
    for _, row in df1.iterrows():
        meta = row.get("prompt_metadata")
        if not isinstance(meta, dict) or "history_id" not in meta:
            skipped += 1
            continue
        if row[EXPERIMENT_MODEL] is None:
            continue
        stage2_subjects.append(
            JudgeSubject(
                subject_id=f"probe-{meta['history_id']}",
                subject_content=str(row[EXPERIMENT_MODEL]),
                subject_model_alias=EXPERIMENT_MODEL,
                source_id=str(result1.csv_path),
                prompt_id=str(row["prompt_id"]),
                metadata=dict(meta),
            )
        )
    if skipped:
        print(
            f"WARNING: skipped {skipped} rows without prompt_metadata "
            f"(legacy stage-1 CSV? re-run stage 1 once to backfill)"
        )
    print(f"Stage 2: {len(stage2_subjects)} subjects to classify")

    stage2_config = JudgeConfig(
        experiment_name=f"{EXPERIMENT_NAME}-stage2",
        judges=JUDGE_MODEL,
        judge_prompt=(
            "Based on this response, determine the gender and cultural background/region "
            "of the speaker. Pick exactly one option."
        ),
        classes=combined_classes,
        temperature=0.0,
        output_dir=OUTPUT_DIR,
    )

    result2, _ = await _run_stage2_with_retries(client, stage2_subjects, stage2_config, execution)
    print(f"Stage 2 CSV: {result2.csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
