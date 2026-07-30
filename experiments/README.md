# experiments

Experiment harnesses that use the `src/inference` layer to run prompt × model matrices
and evaluate the responses. Each subdirectory is a self-contained experiment with its own
runners (`run_*.py` / `.ipynb`), evaluation notebooks (`eval_*.ipynb`), and result data.

## Subdirectories

| Directory | What it does |
|-----------|--------------|
| [`behavioral_audit/`](behavioral_audit/README.md) | Two-stage audit: a subject model answers a persona's job/college question (stage 1), then a judge classifies the recommendation (stage 2). Evaluates whether persona gender/region shifts the advice. Includes frequency-table, significance-test, and LaTeX-table builders. |
| [`direct_probing/`](direct_probing/README.md) | Asks a subject model to directly infer the user's gender + region from the conversation, then a judge scores the guess. Includes the per-axis NONE postprocessing (`postprocess_none.ipynb`) and the refusal-verdict data in `_refusal_filter/`. |
| [`internal_representation/`](internal_representation/README.md) | Linear probes on hidden states (RQ1 internal probing). Modal runners and committed `results_modal_*` plots/metrics. |
| [`judge_audit/`](judge_audit/README.md) | Human-rater validation of the stage-2 judge on a stratified 500-row sample (exact Cohen's κ ≈ 0.70, error concentration, gender/region checks). |

## Loose notebooks

- `eval_real_conversations_full.ipynb` — full analysis over the 50 real WildChat histories (25 male / 25 female, explicit gender labels).

## Conventions

- **Runners** (`run_*.py`) call the inference layer and write raw stage-1 / stage-2 CSVs.
  **Eval notebooks** (`eval_*.ipynb`) read those CSVs and render figures/tables — they do
  not re-run inference.
- Result data lives under each experiment's `results_*/` (committed) and gitignored `logs/`.
- Dependencies come from the repo root `pyproject.toml`: `uv sync` covers the runners and
  eval notebooks (`jupyterlab` included). Activation probes need
  `uv sync --extra internal-rep`; Modal deploy needs `uv sync --extra modal`.

## Related

- [`../src/inference/`](../src/inference/) — the inference layer these experiments call.
- [`../src/generate_backgrounds/`](../src/generate_backgrounds/README.md) — generates the persona conversation histories the runners consume.
- [`../finalresults.ipynb`](../finalresults.ipynb) — report figures for RQ1 and RQ2 from committed results.
- [repo root README](../README.md) — research overview and reproducibility.
