# judge_audit

Human check of the stage-2 judge (`gpt-4o-mini_paid`) used in
[`behavioral_audit`](../behavioral_audit/README.md) and
[`direct_probing`](../direct_probing/README.md).

Three raters independently label a stratified **500-row** sample. We compare the
judge label to their consensus with **exact string match**, and report
**Cohen's κ**. Full analysis:
[`judge_audit_analysis.ipynb`](judge_audit_analysis.ipynb).

See the [experiments overview](../README.md).

## What the judge does

It maps each subject-model reply to one fixed taxonomy label:

| Question | Labels |
|----------|--------|
| q1 (job) | 43 ISCO-08 sub-major groups |
| q2 (college major) | 29 ISCED-F narrow fields |
| direct_probe | `Gender - Region` strings |

Downstream frequency tables use those labels, so this audit bounds how much to
trust them.

## Result

| Metric | Value |
|--------|-------|
| Cohen's κ (exact) | **0.70** [0.66, 0.74] |
| Exact agreement | 72.1% of 491 scored rows |

Nine judge abstentions (`none_declared`) are excluded from κ. Exact-match error
is not significantly tied to persona gender (Fisher p = 0.056) or region
(χ² p = 0.53). Disagreements cluster in a few categories — see notebook §5.

κ is chance-corrected exact agreement:
$\kappa = (p_o - p_e)/(1 - p_e)$, with a percentile bootstrap CI (5,000
resamples, seed **42**, same as the sample draw).

## How coding works

1. Each of three raters picks one taxonomy label.
2. `consensus_label` = mode of the three.
3. Agreement = `final_class == consensus_label` (exact match only).

## Sample

500 rows from 23,989 judged research responses, stratified within question on
`run_tag | subject_model_alias | true_gender | true_region`. q1 and q2 are
equal-weighted; direct-probe is proportional. Gender/region drift vs the
population is ≤2.4pp / ≤0.3pp. Details:
[`judge_audit_sample_meta.json`](judge_audit_sample_meta.json).

## Files

| File | Role |
|------|------|
| [`judge_audit_sample_500.csv`](judge_audit_sample_500.csv) | **Canonical** annotated sample (use this for analysis) |
| [`judge_audit_human_50.csv`](judge_audit_human_50.csv) / [`judge_audit_human_450.csv`](judge_audit_human_450.csv) | Same 500 rows as review sheets (50 + 450) |
| [`judge_audit_analysis.ipynb`](judge_audit_analysis.ipynb) | κ, error concentration, gender/region plots |
| [`prepare_judge_audit_sample.py`](prepare_judge_audit_sample.py) | Draw / refresh the stratified sample |
| [`build_human_review.py`](build_human_review.py) | Build the 50/450 review sheets from the sample |
| [`audit_human_review.py`](audit_human_review.py) | Schema + integrity checks (non-zero exit on failure) |
| [`judge_audit_sample_meta.json`](judge_audit_sample_meta.json) | Stratification diagnostics |

## Key columns

Use these on `judge_audit_sample_500.csv` (and the matching fields on the human
sheets):

| Column | Meaning |
|--------|---------|
| `final_class` | Judge label |
| `none_declared` | Judge abstention (excluded from κ) |
| `rater1_label` … `rater3_label` | Each rater's label |
| `consensus_label` | Mode of the three raters |
| `n_raters` | Always 3 |

On the human sheets only, `rev_1`…`rev_3` mirror `rater*_accepted`, and
`consensus` / `human_best_label` mirror `consensus_label`. Prefer the
`rater*` / `consensus_label` names. `rater*_accepted` / `judge_accepted` are
legacy flags — **not** the headline KPI.

## Reproduce

```bash
# Only if the research population changed
uv run python experiments/judge_audit/prepare_judge_audit_sample.py

uv run python experiments/judge_audit/build_human_review.py
uv run python experiments/judge_audit/audit_human_review.py
uv run python -m jupyter nbconvert --to notebook --execute --inplace \
  experiments/judge_audit/judge_audit_analysis.ipynb
```

The notebook finds the repo root automatically and shows figures inline.
