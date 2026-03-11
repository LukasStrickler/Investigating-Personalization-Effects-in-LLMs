# uv

Short overview of [uv](https://docs.astral.sh/uv/) and how this project uses it.

## What is uv?

uv is a fast Python package and project manager by Astral. It handles virtual environments, dependency resolution, and locking so that installs are reproducible and quick.

This project uses uv as the only supported way to install and run code. You need Python 3.10+ and uv installed.

## Why we use it

- **One command setup**: `uv sync` creates the virtual environment (if missing) and installs all dependencies from `pyproject.toml` and `uv.lock`.
- **Reproducible installs**: The lockfile (`uv.lock`) pins exact versions so everyone gets the same environment.
- **Speed**: Resolves and installs dependencies much faster than pip in typical use.

## Commands you need

| Goal | Command |
|------|---------|
| Install everything (creates `.venv` if needed) | `uv sync` |
| Install with dev tools (tests, lint, type-check) | `uv sync --extra dev` |
| Run a script without activating the venv | `uv run python scripts/your_script.py` |
| Add a dependency | `uv add <package>` |
| Update the lockfile after editing `pyproject.toml` | `uv lock` |

## Workflow

1. Install uv and ensure Python 3.10+ is available (e.g. `uv python install 3.12`).
2. From the project root, run `uv sync`.
3. Activate the environment when you want to use the project in your shell: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`). Or use `uv run` so uv uses `.venv` automatically.

See [Official uv documentation](https://docs.astral.sh/uv/) for more.
