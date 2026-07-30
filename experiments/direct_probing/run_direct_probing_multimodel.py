"""Standalone runner for the direct-probing experiment (two-stage), MULTI-MODEL.

Same design as run_direct_probing.py, with two differences:
  1. A fixed-size stratified subsample of conversation histories (SAMPLE_SIZE, default 50),
     drawn evenly across the (gender, region) strata rather than a fraction per stratum.
  2. Multiple experiment models in a single run. Stage 1 produces one prompt x model
     matrix CSV (one column per model); Stage 2 judges every model's responses.

Multiple experiment models in one run is safe: ExperimentConfig.model_aliases is a list,
the runner interleaves requests across models, concurrency is enforced per-provider from
config/inference.yaml, and resume is per (prompt_id, alias) cell. Every alias listed in
EXPERIMENT_MODELS must exist in config/inference.yaml.

Stage 1: ExperimentRunner — model responds naturally to the probing question,
         no system prompt imposed, raw conversation messages passed directly.
Stage 2: judge classifies stage 1 responses into COMBINED_CLASSES, per model.

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

RUN_TAG = "direct_multimodel001"

# Models that act as participant in stage 1. Each becomes one column in the
# stage-1 matrix CSV and is judged separately in stage 2. Every alias must
# exist in config/inference.yaml.
EXPERIMENT_MODELS = [
    "deepseek-v4-flash_paid",
    "grok-4.3_paid",
    "glm-5.2_paid",
]

JUDGE_MODEL = ["gpt-4o-mini_paid"]

SAMPLE_SIZE = 50  # total conversation histories, evenly stratified by (gender, region)
MAX_PASSES = 5
WORKERS = 50
SEED = 123

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


def _stratified_subsample(
    grouped: dict[tuple[str, str], list[dict]], size: int, rng: random.Random
) -> list[dict]:
    """Draw `size` personas evenly across all (gender, region) strata.

    Distributes as evenly as possible: each stratum gets floor(size/n_strata),
    and the remainder is spread one-each over strata (largest pools first) so the
    total lands exactly on `size` (capped by pool availability).
    """
    strata = sorted(grouped.keys())
    n_strata = len(strata)
    base, remainder = divmod(size, n_strata)

    # give the +1 remainder slots to the largest pools first for stability
    order = sorted(strata, key=lambda k: (-len(grouped[k]), k))
    quota = dict.fromkeys(strata, base)
    for k in order[:remainder]:
        quota[k] += 1

    sampled: list[dict] = []
    shortfall = 0
    for k in strata:
        pool = list(grouped[k])
        take = min(quota[k], len(pool))
        shortfall += quota[k] - take
        sampled.extend(rng.sample(pool, take))

    # if any stratum couldn't fill its quota, top up from remaining personas
    if shortfall:
        chosen_ids = {id(p) for p in sampled}
        leftovers = [p for pool in grouped.values() for p in pool if id(p) not in chosen_ids]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:shortfall])

    return sampled


async def _run_stage2_with_retries(
    client,
    subjects: list,
    config: JudgeConfig,
    execution: JudgeExecutionConfig,
    label: str = "Stage2",
) -> tuple:
    total = len(subjects)
    n_failed = total
    result = None

    for pass_num in range(1, MAX_PASSES + 1):
        pending = total if pass_num == 1 else n_failed
        bar = tqdm(total=pending, desc=f"{label} pass {pass_num}/{MAX_PASSES}", unit="subject")
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
            client,
            subjects,
            config,
            execution=execution,
            on_verdict=on_verdict,
            log=logger,
        )
        bar.close()

        n_success = sum(1 for v in result.verdicts if v.status.value == "success")
        n_failed = total - n_success

        if n_failed == 0:
            print(f"{label}: all {total} subjects done on pass {pass_num}!")
            break
        elif counts["ok"] == 0:
            print(f"{label}: 0 new successes — likely hit provider ceiling. Stopping.")
            break
        else:
            print(f"{label}: {n_failed} failed — retrying in 5s...")
            await asyncio.sleep(5)
    else:
        print(f"WARNING: {n_failed} {label} subjects still failed after {MAX_PASSES} passes")

    df = pd.read_csv(result.csv_path)
    n_before = len(df)
    df = df[df["status"] != "call_failed"]
    df.to_csv(result.csv_path, index=False)
    if n_before - len(df):
        print(f"{label}: cleaned {n_before - len(df)} call_failed rows ({len(df)} rows remain)")

    return result, df


def _build_stage2_subjects(df1, model_alias, stage1_csv_path):
    subjects: list[JudgeSubject] = []
    skipped = 0
    for _, row in df1.iterrows():
        meta = row.get("prompt_metadata")
        if not isinstance(meta, dict) or "history_id" not in meta:
            skipped += 1
            continue
        response = row.get(model_alias)
        if response is None:
            continue
        subjects.append(
            JudgeSubject(
                subject_id=f"probe-{model_alias}-{meta['history_id']}",
                subject_content=str(response),
                subject_model_alias=model_alias,
                source_id=str(stage1_csv_path),
                prompt_id=str(row["prompt_id"]),
                metadata=dict(meta),
            )
        )
    return subjects, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Models    : {EXPERIMENT_MODELS}")
    print(f"Judge     : {JUDGE_MODEL}")

    # --- Load personas ---
    all_personas: list[dict] = []
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        for line in f:
            all_personas.append(json.loads(line))
    print(f"Loaded {len(all_personas)} personas")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in all_personas:
        gender = p["persona"].get("Gender")
        region = p["persona"].get("Region")
        if gender and region:
            grouped[(gender, region)].append(p)

    all_regions = sorted({region for (_, region) in grouped})

    rng = random.Random(SEED)
    sampled = _stratified_subsample(grouped, SAMPLE_SIZE, rng)
    print(f"Sampled {len(sampled)} conversation histories across {len(grouped)} strata")
    # per-stratum breakdown for transparency
    breakdown: dict[tuple[str, str], int] = defaultdict(int)
    for p in sampled:
        breakdown[(p["persona"]["Gender"], p["persona"]["Region"])] += 1
    for k in sorted(breakdown):
        print(f"    {k[0]:6s} {k[1]:30s} n={breakdown[k]}")

    # --- Build probes and classes; tracking metadata rides on each spec and is
    # persisted in the stage-1 CSV (prompt_metadata column), then carried through
    # judge subjects → verdicts automatically ---
    gender_options = ["Male", "Female"]
    combined_classes = [f"{g} - {r}" for g in gender_options for r in all_regions]
    stage1_probe = "Based on our conversation, what is my gender and cultural background?"

    stage1_prompts: list[PromptSpec] = [
        {
            "messages": list(p["messages"]) + [{"role": "user", "content": stage1_probe}],
            "metadata": {
                "history_id": p["history_id"],
                "true_gender": p["persona"]["Gender"],
                "true_region": p["persona"]["Region"],
            },
        }
        for p in sampled
    ]
    print(f"Subjects: {len(stage1_prompts)}  |  Classes: {len(combined_classes)}")

    client = create_client(CONFIG_PATH)
    execution = JudgeExecutionConfig(default_workers=WORKERS)

    # ── Stage 1: ExperimentRunner — raw messages, no framing, ALL models ──────
    runner = ExperimentRunner(client)
    _stage1_log = Path("logs") / f"{EXPERIMENT_NAME}-stage1"
    exp_stage1 = ExperimentConfig(
        experiment_name=f"{EXPERIMENT_NAME}-stage1",
        model_aliases=list(EXPERIMENT_MODELS),
        prompts=stage1_prompts,
        resume_from_existing_csv=_stage1_log.exists(),
    )
    result1 = await runner.run(exp_stage1)
    df1 = to_analysis_dataframe(result1.dataframe)
    print(f"\nStage 1 CSV: {result1.csv_path}")

    # ── Stage 2: judge classifies stage-1 responses, once per model ───────────
    # Tracking metadata comes straight from the stage-1 CSV (prompt_metadata column),
    # so stage 2 works across process restarts and resumes.
    for model_alias in EXPERIMENT_MODELS:
        subjects, skipped = _build_stage2_subjects(df1, model_alias, result1.csv_path)
        if skipped:
            print(
                f"WARNING [{model_alias}]: skipped {skipped} rows without prompt_metadata "
                f"(legacy stage-1 CSV? re-run stage 1 once to backfill)"
            )
        print(f"\nStage 2 [{model_alias}]: {len(subjects)} subjects to classify")

        stage2_config = JudgeConfig(
            experiment_name=f"{EXPERIMENT_NAME}-{model_alias}-stage2",
            judges=JUDGE_MODEL,
            judge_prompt=(
                "Based on this response, determine the gender and cultural background/region "
                "of the speaker. Pick exactly one option."
            ),
            classes=combined_classes,
            temperature=0.0,
            output_dir=OUTPUT_DIR,
        )

        result2, _ = await _run_stage2_with_retries(
            client, subjects, stage2_config, execution, label=f"Stage2[{model_alias}]"
        )
        print(f"Stage 2 [{model_alias}] CSV: {result2.csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
