# examples

Runnable example notebooks demonstrating each layer of the project. Good starting points
before diving into the full experiments.

| Notebook | Shows |
|----------|-------|
| [inference_example.ipynb](inference_example.ipynb) | Runtime layer — single completion, batch processing, error handling. |
| [experiments_example.ipynb](experiments_example.ipynb) | Experiment layer — goal-driven guide to prompt × model matrices. |
| [llm_judge_example.ipynb](llm_judge_example.ipynb) | Judge layer — evaluating model outputs with LLM-as-a-judge. |

Setup: `uv sync` from the repo root, then add API keys to `.env` (see the
[repo root README](../README.md)). Architecture context is in
[docs/architecture.md](../docs/architecture.md).
