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

## Architecture

The project has three inference layers. Start with the runtime layer when you
need direct model calls, use the experiments layer for prompt x model matrices,
and use the judge layer to evaluate responses.

```mermaid
flowchart TD
    Runtime[Runtime inference layer]
    Experiments[Experiment layer]
    Judges[Judge layer]
    Artifacts[CSV / JSONL outputs]

    Runtime --> Experiments
    Runtime --> Judges
    Experiments --> Judges
    Experiments --> Artifacts
    Judges --> Artifacts
```

| Layer | Use it for | Main inputs | Main outputs |
| --- | --- | --- | --- |
| Runtime `inference` | One-off calls, custom scripts, resumable batches | `config/inference.yaml`, model aliases, `InferenceRequest` | `InferenceResult`, runtime logs, batch checkpoints |
| Experiments `inference.experiments` | Prompt x model matrices | Prompt specs, model aliases, `ExperimentConfig` | Experiment CSV, raw dataframe, analysis dataframe |
| Judges `inference.judges` | LLM-based evaluation and classification | Judge subjects, judge aliases, `JudgeConfig` | Judgment CSV, verdict dataframe |

Typical data flow:

1. Configure providers and model aliases in `config/inference.yaml`.
2. Use runtime calls directly, or let the experiment and judge layers call the
   runtime for you.
3. Run experiments to create durable prompt x model CSV matrices.
4. Convert successful experiment cells into judge subjects when you need
   evaluation.
5. Analyze experiment and judgment dataframes in notebooks or scripts.

Optional persona/background generation lives upstream of experiments. It creates
conversation histories that can be passed into experiment prompt specs as
multi-turn context.

Read **[Architecture Overview](docs/architecture.md)** for the full code and
data-flow map, including how model aliases, experiment cells, resume behavior,
and judge verdicts fit together.

## Documentation

- **[Architecture Overview](docs/architecture.md)** - How the runtime, experiments, judges, and data artifacts connect
- **[Experiments Usage Guide](docs/experiments-usage.md)** - How to run prompt x model matrices and analyze results
- **[Provider Configuration](config/inference.example.yaml)** - Example provider, alias, retry, rate-limit, and output-path config
- **[Background Generation](src/generate_backgrounds/README.md)** - Optional persona conversation-history generation

## Examples

See [`examples/`](examples/) for comprehensive usage examples:

**Example Notebooks (Start Here)**
- [`inference_example.ipynb`](examples/inference_example.ipynb) - Runtime layer: single completion, batch processing, error handling
- [`experiments_example.ipynb`](examples/experiments_example.ipynb) - Experiments layer: prompt x model matrices, resume/extend, scheduling controls
- [`llm_judge_example.ipynb`](examples/llm_judge_example.ipynb) - Judge layer: subjects, judge configs, resume, verdict dataframes

Recommended onboarding order:

1. Read [Architecture Overview](docs/architecture.md).
2. Run [`examples/inference_example.ipynb`](examples/inference_example.ipynb).
3. Run [`examples/experiments_example.ipynb`](examples/experiments_example.ipynb).
4. Run [`examples/llm_judge_example.ipynb`](examples/llm_judge_example.ipynb) when you need evaluation.

Default output locations:

- Runtime logs: `logs/inference.jsonl`
- Batch checkpoints: `checkpoints/batch.jsonl`
- Experiment matrices: `logs/<experiment_name>/<timestamp>.csv`
- Judge verdicts: `logs/judges/<experiment_name>.judgments.csv`

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
