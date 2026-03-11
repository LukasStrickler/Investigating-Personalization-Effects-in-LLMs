# Investigating-Personalization-Effects-in-LLMs

A team research project at the University of Mannheim examining whether LLMs infer user identity from conversation history and whether those inferences alter downstream advice, recommendation, and other high-impact responses.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — see [docs/uv.md](docs/uv.md) for a short overview.

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env and add your API keys

# Run a quick test
uv run python examples/smoke_test.py
```

## Documentation

- **[API Usage Guide](docs/inference-usage.md)** - How to use the inference library
- **[Provider Configuration](config/inference.example.yaml)** - Example config with all providers

## Examples

See [`examples/`](examples/) for:

**Research Workflows (High-Level)**
- `run_experiment_matrix.py` - Run prompt x model matrices with CSV persistence and DataFrame results. This is the primary entry point for research experiments.

**Low-Level API**
- `smoke_test.py` - Minimal single-request example
- `batch_example.py` - Batch processing with checkpoint/resume
- `basic_usage.ipynb` - Jupyter notebook with interactive examples
## Development

```bash
# Run tests
pytest tests -q

# Type check
mypy src --ignore-missing-imports

# Lint
ruff check .
```
