# uv

Short overview of [uv](https://docs.astral.sh/uv/) and how this project uses it.

## What is uv?

uv is a fast Python package and project manager by Astral. It handles virtual environments, dependency resolution, and locking so that installs are reproducible and quick.

This project uses uv as the only supported way to install and run code locally. You need Python 3.10+ and uv installed. On HPC login nodes without uv, the cluster docs still show a `pip install -e .` fallback.

## Why we use it

- **One command setup**: `uv sync` creates the virtual environment (if missing) and installs dependencies from `pyproject.toml` and `uv.lock`.
- **Reproducible installs**: The lockfile (`uv.lock`) pins exact versions so everyone gets the same environment.
- **Speed**: Resolves and installs dependencies much faster than pip in typical use.
- **Extras**: Heavy stacks (torch, Modal, vLLM) stay optional so the default install stays lean.

## Commands you need

| Goal | Command |
|------|---------|
| Core install (inference, runners, eval notebooks, `finalresults`) | `uv sync` |
| Dev tools (pytest, ruff, mypy) | `uv sync --extra dev` |
| Activation probes (torch / transformers) | `uv sync --extra internal-rep` |
| Modal SDK (deploy + `modal_run.py`) | `uv sync --extra modal` |
| Local/cluster vLLM *server* package | `uv sync --extra vllm` |
| Everything | `uv sync --all-extras` |
| Run without activating the venv | `uv run python …` / `uv run jupyter lab …` |
| Add a dependency | `uv add <package>` |
| Refresh the lockfile after editing `pyproject.toml` | `uv lock` |

## Workflow

1. Install uv and ensure Python 3.10+ is available (e.g. `uv python install 3.12`).
2. From the project root, run `uv sync` (add extras as needed).
3. Either activate `.venv` (`source .venv/bin/activate`) or prefer `uv run` so uv picks `.venv` automatically.

Quick path for report figures:

```bash
uv sync
uv run jupyter lab finalresults.ipynb
```

See [Official uv documentation](https://docs.astral.sh/uv/) for more.
