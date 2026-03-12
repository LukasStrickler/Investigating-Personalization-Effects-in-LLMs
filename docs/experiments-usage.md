# Experiments Usage Guide

Use this guide for normal research workflows in this project.

The `inference.experiments` API is the primary interface for prompt x model
matrices, resume/extend runs, and dataframe-based analysis. The low-level
`inference` package remains the runtime layer underneath experiments.

Primary references in this repo:

- `examples/experiments_example.ipynb` - end-to-end experiments workflows
- `examples/inference_example.ipynb` - low-level runtime usage
- `config/inference.example.yaml` - provider and alias configuration

## Start Here

Use this default order:

1. Prepare prompts (directly or with `build_experiment_grid(...)`).
2. Build an `ExperimentConfig`.
3. Run with `ExperimentRunner`.
4. Analyze with dataframe helpers.
5. Resume or extend from CSV when needed.

Use low-level runtime calls directly only when you need one-off requests,
custom orchestration, or batch pipelines outside matrix semantics.

## Setup

```bash
uv sync
cp config/inference.example.yaml config/inference.yaml
cp .env.example .env
```

Then set the API keys referenced by your selected provider config.

The example config is notebook-friendly:

- `openrouter` is preconfigured for free models used in examples
- `mock` supports local runs without paid provider calls
- `log_path` defaults to `logs/inference.jsonl`
- `checkpoint_path` defaults to `checkpoints/`

## Quick Start: Run One Experiment

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

    result = await ExperimentRunner(client).run(config)
    print(result.summary.completed_cells, result.summary.total_cells)
    print(result.csv_path)


asyncio.run(main())
```

`run(...)` returns an `ExperimentResult`:

- `dataframe`: full raw matrix dataframe
- `csv_path` and `csv_name`: persisted result file
- `summary`: aggregate counters (`total_cells`, `failed_cells`, etc.)

## Build Prompt x Model Grids

Use `build_experiment_grid(...)` when you want matrix generation logic in one
place (system variants, static turns, sparse runs).

```python
from inference.experiments import ExperimentConfig, build_experiment_grid

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

- rows = full prompt specs (`str` or dict prompt spec)
- columns = model aliases
- sparse runs use `run_cells`; non-selected cells become `not_requested`
- `system_prompt_by_model` is applied at run time (not embedded per row)

## Resume and Extend Existing Runs

Set `resume_from_existing_csv=True` to continue the latest CSV for an
experiment name, or set `existing_csv_path` to resume a specific file.

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
- reruns missing, failed, and rate-limited cells
- appends rows for newly added prompts
- trims rows for removed prompts
- returns loaded results directly if no work remains

## Analyze Results with DataFrame Helpers

Import surface:

```python
from inference.experiments import (
    build_dataframe_from_csv,
    filter_experiment_dataframe,
    to_analysis_dataframe,
)
```

Helper contract:

| Helper | Purpose |
| --- | --- |
| `build_dataframe_from_csv(csv_path)` | Load raw experiment CSV into standard schema |
| `filter_experiment_dataframe(raw_df, ...)` | Filter rows/model columns by completion or content |
| `to_analysis_dataframe(raw_df, ...)` | Convert raw cell dicts to response-text analysis columns |

```python
raw_df = build_dataframe_from_csv(result.csv_path)
complete_df = filter_experiment_dataframe(raw_df, all_complete=True)
analysis_df = to_analysis_dataframe(complete_df)
```

Dataframe shapes:

- raw dataframe columns: `prompt_id`, `prompt`, then one per model alias
- raw model cell: dict with `status`, `response`, `error_message`, `metadata`
- analysis dataframe: `prompt_id`, `prompt`, and one response-text column per
  model alias (`None` for non-success cells)

## ExperimentConfig Reference

`ExperimentConfig` fields:

| Field | Type | Notes |
| --- | --- | --- |
| `experiment_name` | `str` | Required. Labels output path and metadata. |
| `model_aliases` | `list[str]` | Required. Model columns in the matrix. |
| `prompts` | `list[str \| dict[str, Any]]` | Required. Rows in the matrix. Dict prompts must contain `user` or `messages`. |
| `default_system_prompt` | `str \| None` | Optional shared system instruction. |
| `system_prompt_by_model` | `dict[str, str] \| None` | Optional model-specific system instruction override. |
| `run_cells` | `set[tuple[str, str]] \| None` | Optional sparse selector `(prompt_id, model_alias)`. |
| `tools` | `list[dict[str, Any]] \| None` | Optional tool definitions injected into each request. |
| `tool_choice` | `str \| dict[str, Any] \| None` | Optional tool selection policy. |
| `retry` | `ExperimentRetryOptions` | Retry controls (`max_retries`, `base_delay`, `max_delay`). |
| `scheduling` | `ExperimentSchedulingOptions` | Matrix scheduling policy and retry-after guardrail. |
| `verbosity` | `"quiet" \| "normal" \| "verbose" \| "debug"` | Console output level. |
| `resume_from_existing_csv` | `bool` | Resume mode toggle. |
| `existing_csv_path` | `Path \| None` | Explicit CSV path for resume mode. |

Nested options:

`ExperimentRetryOptions`:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `max_retries` | `int` | `3` | Must be `>= 0` |
| `base_delay` | `float` | `1.0` | Must be `> 0` |
| `max_delay` | `float` | `60.0` | Must be `> 0` and `>= base_delay` |

`ExperimentSchedulingOptions`:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `interleave_model_aliases` | `bool` | `True` | `True` = non-blocking interleaving across aliases |
| `max_retry_after_wait_seconds` | `float` | `3600.0` | Max accepted provider retry-after wait before terminal `rate_limited` |

Prompt shapes supported by `prompts`:

```python
# String prompt
"What is machine learning?"

# System + user
{"system": "Answer concisely.", "user": "What is machine learning?"}

# System + full message history
{
    "system": "You are a helpful tutor.",
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+3?"},
    ],
}
```

## How Experiments Run on the Runtime Layer

`ExperimentRunner` builds per-cell `InferenceRequest` objects and calls the
runtime client (`UnifiedInferenceClient`) under the hood.

Per-cell request mapping:

- string prompt -> `InferenceRequest(prompt=<text>, system_prompt=default_system_prompt)`
- dict with `system` + `user` -> `prompt=user`, `system_prompt=system`
- dict with `messages` -> `messages=<list>`, `prompt=""`, `system_prompt=system`
- `tools` and `tool_choice` from `ExperimentConfig` are propagated to every
  request
- `system_prompt_by_model` overrides per alias at run time

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

Runtime exceptions raised during execution include `UnknownModelAliasError` and
`InferenceRequestError`.

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
| `per_model_concurrency` | `int` | Per-alias in-flight cap (`0` = provider cap only) |

Top-level config includes `model_aliases`, `default_retry`, `log_path`,
`checkpoint_path`, and `default_provider`.

## Output Artifacts and Statuses

Where results are written:

- experiments: `logs/<experiment_name>/<timestamp>.csv`
- runtime structured logs: `logs/inference.jsonl`
- batch checkpoint default: `checkpoints/batch.jsonl`

Common experiment cell statuses:

- `success`
- `failed`
- `rate_limited`
- `not_requested`

## Development Commands

```bash
pytest tests -q
mypy src --ignore-missing-imports
ruff check .
```
