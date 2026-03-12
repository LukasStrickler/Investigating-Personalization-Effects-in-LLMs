# Experiments Usage Guide

This project is experiments-first.

Use `inference.experiments` for normal research workflows (prompt x model
matrices, resume, and analysis dataframes). The low-level `inference` package is
mainly the runtime layer underneath experiments.

Primary references in this repo:

- `examples/experiments_example.ipynb` - full experiments workflow
- `examples/inference_example.ipynb` - low-level runtime examples
- `config/inference.example.yaml` - provider/model alias config

## Experiments-First Workflow

Use this order by default:

1. Build or generate prompts for your matrix.
2. Create an `ExperimentConfig`.
3. Run with `ExperimentRunner`.
4. Analyze with dataframe helpers.
5. Resume/extend experiments from CSV when needed.

If you only need one-off requests or custom stream-like orchestration, then use
the low-level runtime API directly.

## Quick Start

```python
import asyncio

from inference import create_client
from inference.experiments import ExperimentConfig, ExperimentRunner


async def main() -> None:
    client = create_client("config/inference.yaml")

    config = ExperimentConfig(
        experiment_name="research-comparison",
        model_aliases=["gemma-3-4b", "mock-test"],
        prompts=[
            "What is the capital of Germany?",
            "Explain quantum entanglement in one sentence.",
        ],
    )

    runner = ExperimentRunner(client)
    result = await runner.run(config)

    print(result.summary.completed_cells, result.summary.total_cells)
    print(result.csv_path)


asyncio.run(main())
```

## Installation and Setup

```bash
uv sync
cp config/inference.example.yaml config/inference.yaml
cp .env.example .env
```

The default example config is notebook-friendly:

- `openrouter` is configured for free OpenRouter models used in examples
- `mock` lets you run examples without paid provider calls
- `log_path` defaults to `logs/inference.jsonl`
- `checkpoint_path` defaults to `checkpoints/`

## Experiments API

Import surface:

```python
from inference.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    build_dataframe_from_csv,
    build_experiment_grid,
    filter_experiment_dataframe,
    to_analysis_dataframe,
)
```

### ExperimentConfig

| Field | Type | Notes |
| --- | --- | --- |
| `experiment_name` | `str` | Labels output directories and metadata |
| `model_aliases` | `list[str]` | Matrix columns |
| `prompts` | `list[str \| dict]` | Matrix rows; dict prompts must include `user` or `messages` |
| `default_system_prompt` | `str \| None` | Shared system instruction |
| `system_prompt_by_model` | `dict[str, str] \| None` | Model-specific system instruction |
| `run_cells` | `set[tuple[str, str]] \| None` | Sparse run selector `(prompt_id, model_alias)` |
| `tools` | `list[dict] \| None` | Optional tool definitions injected into each request |
| `tool_choice` | `str \| dict \| None` | Optional tool selection policy |
| `retry` | `ExperimentRetryOptions` | Experiment-level retry behavior |
| `scheduling` | `ExperimentSchedulingOptions` | Interleave and retry-after guardrail behavior |
| `verbosity` | `"quiet" \| "normal" \| "verbose" \| "debug"` | Console verbosity |
| `resume_from_existing_csv` | `bool` | Continue latest CSV for the experiment name |
| `existing_csv_path` | `Path \| None` | Resume from a specific CSV path |

### Prompt Shapes

Simple prompt list:

```python
prompts = ["What is machine learning?"]
```

Structured prompts with system/user:

```python
prompts = [
    {"system": "Answer concisely.", "user": "What is machine learning?"},
]
```

Structured prompts with full message history:

```python
prompts = [
    {
        "system": "You are a helpful tutor.",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What is 3+3?"},
        ],
    },
]
```

### Build Experiment Grids

Use `build_experiment_grid(...)` to generate row prompts and optional sparse
`run_cells` selectors from system prompts, static turns, and final user
messages.

```python
from inference.experiments import build_experiment_grid

grid = build_experiment_grid(
    system_prompts=["You are concise.", "You are thorough."],
    final_user_messages=["What is 2+2?", "What is the capital of France?"],
    model_aliases=["gemma-3-4b", "mock-test"],
)

config = ExperimentConfig(
    experiment_name="system-vs-request",
    model_aliases=grid.model_aliases,
    prompts=grid.prompts,
    run_cells=grid.run_cells,
)
```

Grid contract:

- rows = full prompt specs
- columns = model aliases
- `system_prompt_by_model` is applied at run time (not embedded per row)
- sparse runs use `run_cells`; non-selected cells are written as `not_requested`

### Result Shape

`ExperimentRunner.run(...)` returns `ExperimentResult`.

```python
result = await runner.run(config)
print(result.dataframe)
print(result.csv_path)
print(result.csv_name)
print(result.summary)
```

Raw dataframe and CSV shape:

- rows = prompts
- columns = `prompt_id`, `prompt`, then one column per model alias
- each model cell is a dict with `status`, `response`, `error_message`, `metadata`

Common status values:

- `success`
- `failed`
- `rate_limited`
- `not_requested`

### Resume and Extend Existing Runs

```python
config = ExperimentConfig(
    experiment_name="my-research",
    model_aliases=["gemma-3-4b", "mock-test"],
    prompts=["Prompt 1", "Prompt 2"],
    resume_from_existing_csv=True,
)
```

Resume behavior:

- keeps successful cells
- re-runs missing, failed, and rate-limited cells
- appends rows for newly added prompts
- trims rows for removed prompts
- returns loaded dataframe directly if no work remains

### DataFrame Helpers

| Helper | Purpose |
| --- | --- |
| `build_dataframe_from_csv(csv_path)` | Load raw experiment CSV into the standard schema |
| `filter_experiment_dataframe(raw_df, ...)` | Filter rows/model columns by completion or content |
| `to_analysis_dataframe(raw_df, ...)` | Convert raw cell dicts into prompt + response columns |

```python
from inference.experiments import (
    build_dataframe_from_csv,
    filter_experiment_dataframe,
    to_analysis_dataframe,
)

raw_df = build_dataframe_from_csv(result.csv_path)
complete_df = filter_experiment_dataframe(raw_df, all_complete=True)
analysis_df = to_analysis_dataframe(complete_df)
```

## How Experiments Run on the Inference Runtime

`ExperimentRunner` uses the low-level runtime client for each matrix cell. You
usually do not call these runtime helpers directly unless you need custom
orchestration.

Top-level runtime exports:

```python
from inference import (
    InferenceConfig,
    InferenceRequest,
    InferenceResult,
    __version__,
    create_client,
    run_batch,
    run_completion,
)
```

Low-level request fields include:

- `model_alias`, `prompt`
- optional `system_prompt`, `messages`
- optional `tools`, `tool_choice`
- optional `max_tokens`, `temperature`

Low-level result fields include:

- model identity: `model_alias`, `provider`, `model`
- response: `content`
- usage/latency: `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_ms`, `retry_count`
- optional provider extras: `tool_calls`, `metadata`

`UnknownModelAliasError` and `InferenceRequestError` are runtime exceptions from
client execution. They are not exported via `from inference import *`.

## Provider Configuration (Runtime Layer)

Each provider entry in `config/inference.yaml` supports:

| Field | Type | Notes |
| --- | --- | --- |
| `name` | `openai \| anthropic \| openrouter \| mock` | Provider identifier |
| `api_key_env` | `str` | API key environment variable |
| `rate_limit` | object | Optional RPM/TPM limits |
| `retry` | object | Optional retry override |
| `base_url` | `str \| null` | Custom endpoint override |
| `default_model` | `str \| null` | Provider default model |
| `max_concurrency` | `int` | Max in-flight provider requests (`0` = unlimited) |
| `per_model_concurrency` | `int` | Per-alias concurrency cap (`0` = provider cap only) |

Top-level config fields include `model_aliases`, `default_retry`, `log_path`,
`checkpoint_path`, and `default_provider`.

## Logging, Checkpoints, and Retry

- Structured inference logs are written to `logs/inference.jsonl`.
- `run_batch(...)` uses JSONL checkpoints, typically `checkpoints/batch.jsonl`.
- Checkpoint statuses: `pending`, `success`, `fatal_failure`, `retryable_failure`.
- Retry uses exponential backoff with jitter.
- Experiments honor `ExperimentSchedulingOptions.max_retry_after_wait_seconds`
  before marking a cell terminally `rate_limited`.

## Current Scope

Included:

- prompt x model experiment matrix execution
- CSV resume/extend workflows
- dataframe filtering and analysis helpers
- multi-provider runtime integration with rate limits/retries
- mock provider support for tests/examples

Not included:

- streaming response APIs
- built-in response caching
- dashboard/monitoring UI
- distributed execution

## Development

```bash
pytest tests -q
mypy src --ignore-missing-imports
ruff check .
```
