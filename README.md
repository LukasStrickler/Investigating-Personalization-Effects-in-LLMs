# Investigating-Personalization-Effects-in-LLMs

A team research project at the University of Mannheim examining whether LLMs infer user identity from conversation history and whether those inferences alter downstream advice, recommendation, and other high-impact responses.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — see [docs/uv.md](docs/uv.md) for a short overview.

## Quick Start

```bash
uv sync
cp .env.example .env
# Edit .env and add your API keys
```

## Documentation

- **[Experiments Usage Guide](docs/experiments-usage.md)** — matrices, resume, `prompt_metadata` tracking
- **[Behavioral audit cost estimate](docs/cost-estimate-behavioral-audit.md)** — scope and pricing; calibrate with `experiments/estimate_cost.py`
- **[Provider Configuration](config/inference.example.yaml)** — example config with all providers

## Examples

See [`examples/`](examples/):

- `inference_example.ipynb` — low-level API
- `experiments_example.ipynb` — experiment matrices
- `llm_judge_example.ipynb` — LLM-as-judge

Research runs: `experiments/behavioral_audit.ipynb`, `experiments/direct_probing.ipynb`.

```bash
jupyter lab
```

## Development

```bash
pytest tests -q
mypy src --ignore-missing-imports
ruff check .
```
