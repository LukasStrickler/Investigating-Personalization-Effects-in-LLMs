# experiments

Experiment harnesses that use the `src/inference` layer to run prompt × model matrices
and evaluate the responses. Each subdirectory is a self-contained experiment with its own
runners (`run_*.py` / `.ipynb`), evaluation notebooks (`eval_*.ipynb`), and result data.

## Subdirectories

| Directory | What it does |
|-----------|--------------|
| [`behavioral_audit/`](behavioral_audit/README.md) | Two-stage audit: a subject model answers a persona's job/college question (stage 1), then a judge classifies the recommendation (stage 2). Evaluates whether persona gender/region shifts the advice. Includes frequency-table, significance-test, and LaTeX-table builders. |
| [`direct_probing/`](direct_probing/README.md) | Asks a subject model to directly infer the user's gender + region from the conversation, then a judge scores the guess. Includes the per-axis NONE postprocessing (`postprocess_none.ipynb`) and the refusal-verdict data in `_refusal_filter/`. |
| [`judge_audit/`](judge_audit/) | Human-validation harness for the LLM judge: samples judged rows and builds review sheets to measure judge accuracy against human labels. |

## Loose notebooks

- `eval_real_conversations_full.ipynb` — full analysis over the 50 real WildChat histories (25 male / 25 female, explicit gender labels).

## Conventions

- **Runners** (`run_*.py`) call the inference layer and write raw stage-1 / stage-2 CSVs.
  **Eval notebooks** (`eval_*.ipynb`) read those CSVs and render figures/tables — they do
  not re-run inference.
- Result data lives under each experiment's `results_*/` (committed) and gitignored `logs/`.
- Dependencies come from the repo root `pyproject.toml`: `uv sync` covers the runners and
  eval notebooks.

## Related

- [`../src/inference/`](../src/inference/) — the inference layer these experiments call.
- [`../src/generate_backgrounds/`](../src/generate_backgrounds/README.md) — generates the persona conversation histories the runners consume.
- [repo root README](../README.md) — architecture overview and setup.
