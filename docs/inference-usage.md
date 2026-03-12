# Inference Library Usage Guide

This guide covers the current public API of the `inference` package and its
high-level `inference.experiments` layer.

Use it together with these repo examples:

- `examples/inference_example.ipynb` - low-level API, batch runs, logging, errors
- `examples/experiments_example.ipynb` - experiment matrices, resume, filtering, analysis
- `config/inference.example.yaml` - current config schema and example aliases

## Installation

```bash
uv sync
```

## Configuration

Copy the example config and set the environment variables referenced by your
enabled providers:

```bash
cp config/inference.example.yaml config/inference.yaml
cp .env.example .env
```

The example config is intentionally notebook-friendly:

- `openrouter` is configured for the free OpenRouter models used in the examples
- `mock` lets you run examples without a real API call
- `log_path` defaults to `logs/inference.jsonl`
- `checkpoint_path` defaults to `checkpoints/`

### Provider schema

Each provider entry supports these fields:

| Field | Type | Notes |
| --- | --- | --- |
| `name` | `openai \| anthropic \| openrouter \| mock` | Supported provider identifier |
| `api_key_env` | `str` | Environment variable containing the API key |
| `rate_limit` | object | Optional RPM/TPM limits |
| `retry` | object | Optional provider-specific retry override |
| `base_url` | `str \| null` | Optional custom endpoint |
| `default_model` | `str \| null` | Optional provider default |
| `max_concurrency` | `int` | Max in-flight requests for the provider; `0` means unlimited |
| `per_model_concurrency` | `int` | Max in-flight requests per alias; `0` falls back to provider concurrency |

### Example config

See `config/inference.example.yaml` for the full schema. The current example
looks like this:

```yaml
providers:
  openrouter:
    name: openrouter
    api_key_env: OPENROUTER_API_KEY
    rate_limit:
      requests_per_minute: 20
      tokens_per_minute: 100000
    retry:
      max_retries: 3
      base_delay: 2.0
      max_delay: 120.0
    base_url: https://openrouter.ai/api/v1
    default_model: google/gemma-3-4b-it:free

  mock:
    name: mock
    api_key_env: MOCK_API_KEY
    rate_limit:
      requests_per_minute: 0
      tokens_per_minute: 0
    retry:
      max_retries: 1
      base_delay: 0.1
      max_delay: 1.0
    default_model: mock-model

model_aliases:
  gemma-3-4b:
    alias: gemma-3-4b
    provider: openrouter
    model: google/gemma-3-4b-it:free

  mock-test:
    alias: mock-test
    provider: mock
    model: mock-model

default_retry:
  max_retries: 2
  base_delay: 0.5
  max_delay: 30.0

log_path: logs/inference.jsonl
checkpoint_path: checkpoints/
default_provider: openrouter
```

## Low-Level API

The top-level package exports a small public surface:

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

`UnknownModelAliasError` and `InferenceRequestError` are part of the runtime
behavior of the low-level client, but they are not exported through
`from inference import *`.

### Create a client

`create_client(config_path)` loads and validates the YAML config, configures the
provider adapter, and returns a `UnifiedInferenceClient`.

```python
from inference import create_client

client = create_client("config/inference.yaml")
```

### Run a single completion

All inference calls are async.

```python
import asyncio

from inference import InferenceRequest, create_client, run_completion


async def main() -> None:
    client = create_client("config/inference.yaml")
    request = InferenceRequest(
        model_alias="gemma-3-4b",
        prompt="Summarize the purpose of this repository in one sentence.",
    )
    result = await run_completion(client, request)
    print(result.content)


asyncio.run(main())
```

### `InferenceRequest`

The low-level request type now supports plain prompts, chat-style messages, and
tool-calling fields.

| Field | Type | Notes |
| --- | --- | --- |
| `model_alias` | `str` | Alias from `model_aliases` in the YAML config |
| `prompt` | `str` | Base prompt text; still required by the current request contract |
| `system_prompt` | `str \| None` | Optional system instruction |
| `messages` | `list[dict] \| None` | Optional chat-style message list |
| `tools` | `list[dict] \| None` | Optional tool definitions passed through to the provider |
| `tool_choice` | `str \| dict \| None` | Optional tool selection policy |
| `max_tokens` | `int \| None` | Optional generation limit |
| `temperature` | `float \| None` | Optional sampling temperature |

Example with chat messages and a system prompt:

```python
request = InferenceRequest(
    model_alias="gemma-3-4b",
    prompt="Continue the conversation.",
    system_prompt="You are concise and factual.",
    messages=[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+3?"},
    ],
)
```

### `InferenceResult`

`run_completion(...)` returns an `InferenceResult` with both response content and
execution metadata.

| Field | Type | Notes |
| --- | --- | --- |
| `model_alias` | `str` | Alias from the request |
| `provider` | `str` | Provider that served the request |
| `model` | `str` | Concrete provider model identifier |
| `content` | `str` | Text response |
| `prompt_tokens` | `int \| None` | Prompt token count if reported |
| `completion_tokens` | `int \| None` | Completion token count if reported |
| `total_tokens` | `int \| None` | Total token count if reported |
| `latency_ms` | `float` | Request latency |
| `retry_count` | `int` | Number of retries used before success |
| `tool_calls` | `list[dict] \| None` | Tool-call payloads returned by the provider |
| `metadata` | `dict \| None` | Extra provider metadata |

`metadata` may include values such as `system_prompt_folded` and the folded
`system_prompt` for providers that convert system content into user messages.

### Batch processing

`run_batch(config_path, requests)` creates its own client and runs an async
iterator of `InferenceRequest` values with checkpoint persistence.

```python
from inference import InferenceRequest, run_batch


async def generate_requests():
    yield InferenceRequest(model_alias="gemma-3-4b", prompt="Hello")
    yield InferenceRequest(model_alias="mock-test", prompt="World")


await run_batch("config/inference.yaml", generate_requests())
```

Batch behavior:

- progress is checkpointed after each request
- the checkpoint file defaults to `checkpoints/batch.jsonl`
- structured logs are written to `logs/inference.jsonl`
- rerunning with the same checkpoint skips completed work

## Experiments Layer

Use `inference.experiments` when you want prompt x model matrices, durable CSV
results, resume semantics, and analysis-friendly helpers.

```python
from inference import create_client
from inference.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    build_dataframe_from_csv,
    build_experiment_grid,
    filter_experiment_dataframe,
    to_analysis_dataframe,
)
```

### Quick start

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

### `ExperimentConfig`

`ExperimentConfig` now includes tool-related options and resume settings in the
public dataclass.

| Field | Type | Notes |
| --- | --- | --- |
| `experiment_name` | `str` | Labels outputs and metadata |
| `model_aliases` | `list[str]` | Model columns in the matrix |
| `prompts` | `list[str \| dict]` | Row inputs; each dict must contain `user` or `messages` |
| `default_system_prompt` | `str \| None` | Optional shared system prompt |
| `system_prompt_by_model` | `dict[str, str] \| None` | Per-model system instruction |
| `run_cells` | `set[tuple[str, str]] \| None` | Sparse grid selection by `(prompt_id, model_alias)` |
| `tools` | `list[dict] \| None` | Optional tool definitions passed into each low-level request |
| `tool_choice` | `str \| dict \| None` | Optional tool selection policy |
| `retry` | `ExperimentRetryOptions` | Experiment-level retry defaults |
| `scheduling` | `ExperimentSchedulingOptions` | Interleave policy and retry-after guardrail |
| `verbosity` | `"quiet" \| "normal" \| "verbose" \| "debug"` | Console logging detail |
| `resume_from_existing_csv` | `bool` | Resume the latest matching experiment CSV |
| `existing_csv_path` | `Path \| None` | Explicit CSV to resume instead of auto-resolving |

### Prompt shapes

The experiments layer accepts both simple strings and structured prompt specs.

Simple prompt:

```python
prompts = ["What is machine learning?"]
```

Structured prompt with system and user fields:

```python
prompts = [
    {"system": "Answer concisely.", "user": "What is machine learning?"},
]
```

Structured prompt with full message history:

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

### Build experiment grids

`build_experiment_grid(...)` is the main helper for generating prompt rows from
system prompts, prior turns, final user messages, and sparse-run selections.

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

Important grid rules:

- rows are full prompt specs
- columns are model aliases
- `system_prompt_by_model` is applied at run time, not embedded into the row spec
- sparse runs use `run_cells`; non-selected cells are written as `not_requested`

### Result shape

`ExperimentRunner.run(...)` returns an `ExperimentResult`:

```python
result = await runner.run(config)

print(result.dataframe)
print(result.csv_path)
print(result.csv_name)
print(result.summary)
```

The raw DataFrame and CSV use a universal shape:

- rows = prompts
- columns = `prompt_id`, `prompt`, then one column per model alias
- each model cell is a dict with `status`, `response`, `error_message`, `metadata`

Supported cell statuses include:

- `success`
- `failed`
- `rate_limited`
- `not_requested`

### Resume an existing CSV

Set `resume_from_existing_csv=True` to continue the latest CSV for the same
`experiment_name`.

```python
config = ExperimentConfig(
    experiment_name="my-research",
    model_aliases=["gemma-3-4b", "mock-test"],
    prompts=["Prompt 1", "Prompt 2"],
    resume_from_existing_csv=True,
)
```

When resume mode is enabled:

- successful cells are preserved
- missing, failed, and rate-limited cells are re-run
- new prompts append new rows
- removed prompts are trimmed from the CSV and returned DataFrame
- if no work remains, the runner returns the loaded DataFrame without new API calls

### DataFrame helpers

The experiments package now exports three analysis helpers:

| Helper | Purpose |
| --- | --- |
| `build_dataframe_from_csv(csv_path)` | Load a raw experiment CSV into the universal DataFrame shape |
| `filter_experiment_dataframe(raw_df, ...)` | Filter raw rows and selected model columns by status, prompt text, or completion criteria |
| `to_analysis_dataframe(raw_df, ...)` | Convert raw cell dicts into prompt + response text per model |

Example:

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

## Logging and Checkpoints

### Structured logs

Inference events are appended to `logs/inference.jsonl` by default.

Successful log entries include fields like:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456Z",
  "provider": "openrouter",
  "model": "google/gemma-3-4b-it:free",
  "status": "success",
  "latency_ms": 450.5,
  "prompt_tokens": 10,
  "completion_tokens": 20,
  "total_tokens": 30,
  "retry_count": 0
}
```

Logs are metadata-only. They do not store prompt text or response content.

### Batch checkpoints

`run_batch(...)` uses JSONL checkpoints, typically at `checkpoints/batch.jsonl`.

Checkpoint states include:

- `pending`
- `success`
- `fatal_failure`
- `retryable_failure`

Delete the checkpoint file if you want to restart the batch from scratch.

## Error handling and retries

Low-level inference classifies failures into retry-aware categories such as:

- `RATE_LIMIT`
- `TIMEOUT`
- `NETWORK_ERROR`
- `SERVER_ERROR`
- `AUTH_FAILURE`
- `INVALID_REQUEST`
- `UNKNOWN`

Default retry behavior:

- exponential backoff with jitter
- default low-level retry config of `max_retries=2`, `base_delay=0.5`, `max_delay=30.0`
- provider-specific retry config overrides the defaults

For experiments, `ExperimentSchedulingOptions.max_retry_after_wait_seconds`
controls how long a `Retry-After` delay may be before the cell is marked as
terminally `rate_limited`.

## Current scope

Included:

- multi-provider inference through the supported provider adapters
- per-provider rate limiting and concurrency limits
- structured JSONL logging
- resumable batch execution
- prompt x model experiment matrices
- CSV resume and analysis helpers
- mock provider support for tests and examples

Not included:

- streaming response APIs
- built-in response caching
- dashboard or UI monitoring
- distributed execution

## Development

```bash
pytest tests -q
mypy src --ignore-missing-imports
ruff check .
```
