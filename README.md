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
```

## Documentation

- **[Experiments Usage Guide](docs/experiments-usage.md)** - How to run experiments and analyze results
- **[Provider Configuration](config/inference.example.yaml)** - Example config with all providers

## Examples

See [`examples/`](examples/) for comprehensive usage examples:

**Example Notebooks (Start Here)**
- `inference_example.ipynb` - Complete low-level API guide: single completion, batch processing, error handling
- `experiments_example.ipynb` - Complete experiments layer guide: prompt×model matrices, resume/extend, scheduling controls

**Quick Start**
```bash
# Run a quick test in Jupyter Lab
jupyter lab  # Or use the notebooks directly in VS Code/Jupyter
```
## Development

```bash
# Run tests
pytest tests -q

# Type check
mypy src --ignore-missing-imports

# Lint
ruff check .
```
