# Inference Library Usage Guide

This guide covers how to use the `inference` package for LLM research.

## Installation

```bash
uv sync
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and set your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

### YAML Configuration

See [`config/inference.example.yaml`](../config/inference.example.yaml) for a complete example.

```yaml
providers:
  openai:
    name: openai
    api_key_env: OPENAI_API_KEY
    rate_limit:
      requests_per_minute: 500
      tokens_per_minute: 150000
    retry:
      max_retries: 3
      base_delay: 1.0
      max_delay: 60.0
    default_model: gpt-4o-mini

model_aliases:
  gpt-4o-mini:
    alias: gpt-4o-mini
    provider: openai
    model: gpt-4o-mini

default_retry:
  max_retries: 2
  base_delay: 0.5
  max_delay: 30.0

log_path: logs/inference.jsonl
checkpoint_path: checkpoints/batch.jsonl
```

### Provider Configuration

| Provider | Description | API Docs |
|----------|-------------|----------|
| `openai` | OpenAI GPT models | https://platform.openai.com/docs |
| `anthropic` | Claude models | https://docs.anthropic.com/claude |
| `openrouter` | Multi-model gateway | https://openrouter.ai/docs |
| `mock` | Test-only (no API calls) | — |

The mock provider returns deterministic responses and is useful for testing and CI. It is not for production use.

## Python API

### Basic Usage

```python
from inference import create_client, run_completion, InferenceRequest

# Create a client from config
client = create_client("config/inference.yaml")

# Run a single completion
request = InferenceRequest(
    model_alias="gpt-4o-mini",
    prompt="What is the capital of France?"
)
result = await run_completion(client, request)
print(result.content)
```

### Async Context

All inference operations are async. Use `asyncio.run()` or run in an async context:

```python
import asyncio

async def main():
    client = create_client("config/inference.yaml")
    request = InferenceRequest(model_alias="gpt-4o-mini", prompt="Hello!")
    result = await run_completion(client, request)
    print(result.content)

asyncio.run(main())
```

### Batch Processing

```python
from inference import run_batch, InferenceRequest

async def generate_requests():
    yield InferenceRequest(model_alias="gpt-4o-mini", prompt="Hello")
    yield InferenceRequest(model_alias="claude-3-5-sonnet", prompt="World")

await run_batch("config/inference.yaml", generate_requests())
```

## Experiments Layer

The inference library provides a two-layer architecture:

- **`inference`** - Low-level layer handling providers, retries, rate limits, normalization, and request-level logging
- **`inference.experiments`** - High-level layer for research workflows: experiment naming, matrix orchestration, CSV state management, and aggregate results

Use the experiments layer when you need to run systematic comparisons across multiple prompts and models. Use the low-level API for single requests or custom batch processing.

### Quick Start

```python
from inference import create_client
from inference.experiments import ExperimentConfig, ExperimentRunner

client = create_client("config/inference.yaml")

config = ExperimentConfig(
    experiment_name="my-research",
    model_aliases=["gpt-4o-mini", "claude-3-5-sonnet"],
    prompts=["What is AI?", "Explain machine learning."],
)

runner = ExperimentRunner(client=client)
result = await runner.run(config)

# Access results
print(result.dataframe)  # pandas DataFrame
print(result.csv_path)   # Path to CSV file
```

### ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    experiment_name: str                    # Labels outputs and metadata
    model_aliases: list[str]                # Model columns in the matrix
    prompts: list[str]                      # Row inputs (at least one required)
    retry: ExperimentRetryOptions           # Retry behavior per cell
    scheduling: ExperimentSchedulingOptions # Concurrency and rate-limit handling
    verbosity: Literal["quiet", "normal", "verbose", "debug"]
    resume_from_existing_csv: bool           # Enable recovery from existing CSV
    existing_csv_path: Path | None           # Explicit CSV path (optional)
```

### ExperimentRunner

The runner orchestrates the full prompt x model matrix:

1. Creates CSV at `logs/<experiment-name>/<timestamp>.csv`
2. Executes all N x M cells with retry/scheduling
3. Persists each cell result immediately
4. Returns DataFrame + CSV metadata after full completion

```python
runner = ExperimentRunner(client=client)
result = await runner.run(config)
```

### ExperimentResult

```python
@dataclass
class ExperimentResult:
    dataframe: Any           # pandas DataFrame with full matrix state
    csv_path: Path          # Path to durable CSV
    csv_name: str           # CSV filename
    summary: ExperimentSummary  # Aggregate counters

@dataclass
class ExperimentSummary:
    prompt_count: int
    model_count: int
    total_cells: int
    completed_cells: int
    failed_cells: int
    rate_limited_cells: int
```

### CSV Format and File Paths

**File Path Convention:**
```
logs/<experiment-name>/<timestamp>.csv
```

**CSV Structure:**
- Rows: prompts (in order provided)
- Columns: model aliases (in order provided)
- Cells: response content or status marker

**CSV Status Values:**
- `success` - Cell completed successfully
- `failed` - Cell failed after retry exhaustion
- `rate_limited` - Terminal status for long provider retry-after waits

### Resume and Retry-In-Place

Enable resume to recover from interruptions without reprocessing completed cells:

```python
config = ExperimentConfig(
    experiment_name="my-research",
    model_aliases=["gpt-4o-mini", "claude-3-5-sonnet"],
    prompts=["Prompt 1", "Prompt 2"],
    resume_from_existing_csv=True,  # Retry only failed/rate-limited cells
)
```

When `resume_from_existing_csv=True`:
- The runner locates the existing CSV for this experiment
- Only cells with status `failed`, `rate_limited`, or missing are reprocessed
- Successfully completed cells are preserved

### Retry Defaults and Rate-Limit Handling

**Default Retry Options:**
```python
ExperimentRetryOptions(
    max_retries=3,          # Retry failed cells up to 3 times
    base_delay=1.0,         # Initial delay between retries (seconds)
    max_delay=60.0,         # Maximum delay between retries (seconds)
)
```

**Scheduling Options:**
```python
ExperimentSchedulingOptions(
    max_concurrency=0,                      # 0 = unlimited
    per_model_concurrency=0,              # 0 = unlimited
    interleave_model_aliases=True,        # Round-robin across models
    max_retry_after_wait_seconds=3600.0,   # 1 hour max wait for rate limits
)
```

**Rate-Limit Behavior:**
- Provider returns `Retry-After` header < 1 hour: wait and retry
- Provider returns `Retry-After` header > 1 hour: mark cell as `rate_limited` (terminal)
- Exponential backoff with jitter between retries

### Complete Example

See [`examples/run_experiment_matrix.py`](../examples/run_experiment_matrix.py) for a working example:

```python
import asyncio
from inference import create_client
from inference.experiments import ExperimentConfig, ExperimentRunner

async def main():
    client = create_client("config/inference.yaml")
    
    config = ExperimentConfig(
        experiment_name="research-comparison",
        model_aliases=["gpt-4o-mini", "claude-3-5-sonnet"],
        prompts=[
            "What is the capital of Germany?",
            "Explain quantum entanglement in one sentence.",
        ],
    )
    
    runner = ExperimentRunner(client=client)
    result = await runner.run(config)
    
    print(f"Completed: {result.summary.completed_cells}/{result.summary.total_cells}")
    print(result.dataframe)

asyncio.run(main())
```

### Public API Surface

```python
from inference import (
    create_client,      # Factory for UnifiedInferenceClient
    run_completion,     # Single request helper
    run_batch,          # Batch processing helper
    InferenceRequest,   # Request dataclass
    InferenceResult,    # Result dataclass
    InferenceConfig,     # Config dataclass
)
```

### InferenceRequest

```python
@dataclass
class InferenceRequest:
    model_alias: str       # Alias from config (e.g., "gpt-4o-mini")
    prompt: str             # The prompt text
    max_tokens: int = 1024  # Maximum tokens to generate
    temperature: float = 1.0  # Sampling temperature
```

### InferenceResult

```python
@dataclass
class InferenceResult:
    model_alias: str       # Original alias
    provider: str          # Provider used (e.g., "openai")
    model: str             # Actual model called
    content: str           # Generated text
    prompt_tokens: int     # Tokens in prompt
    completion_tokens: int # Tokens in completion
    total_tokens: int      # Total tokens
    latency_ms: float      # Request latency
    retry_count: int       # Number of retries
```

## Logging

### Log Schema

Structured logs are written as JSONL to `logs/inference.jsonl`:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456Z",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "status": "success",
  "latency_ms": 450.5,
  "prompt_tokens": 10,
  "completion_tokens": 20,
  "total_tokens": 30
}
```

Failed requests include `status: "failure"` and an `error_type`/`error_message` field.

### Checkpoint Behavior

Batch processing checkpoints progress to `checkpoints/batch.jsonl`. If interrupted, re-running resumes from the last completed request.

Checkpoint states:
- `pending` — request is queued
- `success` — completed successfully (terminal)
- `fatal_failure` — failed, not retryable (terminal)
- `retryable_failure` — failed, will retry

Delete the checkpoint file to restart from the beginning.

## Error Handling

### Error Categories

| Category | Retryable | Example |
|----------|-----------|---------|
| `RATE_LIMIT` | Yes | 429 Too Many Requests |
| `TIMEOUT` | Yes | Request timeout |
| `NETWORK_ERROR` | Yes | Connection error |
| `SERVER_ERROR` | Yes | 500 Internal Server Error |
| `AUTH_FAILURE` | No | 401 Invalid API key |
| `INVALID_REQUEST` | No | 400 Bad request |
| `UNKNOWN` | No | Unclassified error |

### Retry Behavior

- Exponential backoff with jitter
- Default: 2 retries, 0.5s base delay, 30s max delay
- Provider-specific retry config overrides defaults

## Rate Limiting

Per-provider RPM (requests per minute) and TPM (tokens per minute) limits:

```yaml
rate_limit:
  requests_per_minute: 500
  tokens_per_minute: 150000
```

Set to `0` for unlimited (used by mock provider).

Rate limits are enforced client-side before making API calls to avoid 429 errors.

## v1 Scope

### Included

- Multi-provider inference (OpenAI, Anthropic, OpenRouter)
- Rate limiting with per-provider quotas
- Exponential backoff with jitter for retries
- Structured JSONL logging
- Batch processing with checkpoint/resume
- Model aliases for convenient reference
- Mock provider for testing

### Excluded (Future Work)

- **Streaming responses** — only complete responses are supported
- **Response caching** — no built-in caching layer
- **Dashboard/monitoring UI** — logs are JSONL only
- **Distributed execution** — single-process only

## Development

```bash
# Run tests
pytest tests -q

# Type check
mypy src --ignore-missing-imports

# Lint
ruff check .
```
