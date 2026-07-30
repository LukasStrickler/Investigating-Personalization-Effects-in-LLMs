# behavioral_audit

Tests whether a subject model's **advice changes with the persona's gender/region**.
Two stages: the subject model answers a persona's question (stage 1), then an LLM judge
classifies the recommendation into a taxonomy (stage 2). Differences in the class
distribution across demographic groups are the audit signal.

See the [experiments overview](../README.md) for context and
[`../../src/inference/`](../../src/inference/) for the inference layer the runners call.
Personas come from [`../../src/generate_backgrounds/`](../../src/generate_backgrounds/README.md).

## Two questions

- **Q1 — jobs**: subject recommends a job → judged into
  `indicator_hierarchy/jobs_classification.json` (sub-major groups → 10 major groups).
- **Q2 — college**: subject recommends a college major → judged into
  `indicator_hierarchy/college_classification.json` (narrow → broad fields).

## Conditions & runners

Runs are tagged by condition. `run_behavioral_audit*.py` produce stage-1 + stage-2 CSVs:

| Runner | Condition |
|--------|-----------|
| `baseline/run_behavioral_audit_baseline*.py` | No-persona baseline (unconditional). |
| `run_behavioral_audit.py` / `_2.py` | Full persona runs (`full001`, `full002`). |
| `run_behavioral_audit_full001_e2b.py`, `_ministral3_8b.py` | Modal-hosted subject models. |
| `run_behavioral_audit_modal.py` | Modal variant. |
| `run_behavioral_audit_wildchat.py` | Real WildChat histories. |
| `status.py` / `status_csv.py` | Progress of an in-flight run. |

## Evaluation & tables

- `baseline/eval_behavioral_audit_baseline.ipynb`, `eval_behavioral_audit_full001*.ipynb`
  — per-condition distributions (overall, per Q1/Q2, per model).
- `eval_comparison_full001_full002.ipynb` — cross-run comparison.
- `eval_none_analysis_q2.ipynb` — analysis of `__NONE__` (unclassifiable) responses.
- `analyse_failures.ipynb` — failed / unparseable judgments.
- **Table builders** (in `significance_test/`): `merge_stage2_judgments.py` →
  `build_frequency_tables.py` → `build_significance_tables.py` → `build_latex_tables.py`
  (orchestrated by `build_behavioral_audit_tables.py`). Produce the frequency,
  FDR-significance, and LaTeX tables under `results_behavioral_audit/results_merged/`.
- `export_results.py` — export merged results.

Rebuild from committed stage-2 CSVs (no model calls). Defaults already point at
`results_behavioral_audit/`:

```bash
uv run python experiments/behavioral_audit/significance_test/build_behavioral_audit_tables.py
uv run python experiments/behavioral_audit/significance_test/build_latex_tables.py
```

## Directories

- `baseline/` — no-persona baseline runners + its eval notebook.
- `indicator_hierarchy/` — the judge taxonomies (`jobs_classification.json`,
  `college_classification.json`) loaded by the runners and eval notebooks.
- `significance_test/` — the table-builder pipeline (merge → frequency → significance
  → LaTeX).
- `results_behavioral_audit/` — **canonical** committed outputs:
  - `results_<tag>/` — per-run stage-1/stage-2 CSVs (`results_full001*`,
    `results_baseline*`, `results_wildchat`, …)
  - `results_merged/` — frequency tables, significance tests, LaTeX
  - `figures/` — eval figures (including `wildchat001-eval/`)
  - `model-comparison-results/` — cross-model comparison notebooks + figures

  This is what [`finalresults.ipynb`](../../finalresults.ipynb) and the table
  builders read/write by default.
