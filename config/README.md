# Config Directory

This directory contains YAML configuration files for the runtime inference
layer. Research code should use model aliases from these files instead of
hard-coding provider model ids.

- `inference.example.yaml` - Main example config for notebooks and local runs.
  It defines providers, model aliases, retry behavior, rate limits, structured
  log paths, checkpoint paths, and the default provider.
- `judge_smoke.yaml` - Small OpenRouter-focused config for the
  `examples/llm_judge_example.ipynb` smoke workflow.

Typical setup:

```bash
cp config/inference.example.yaml config/inference.yaml
cp .env.example .env
```

Then set the API keys referenced by `api_key_env` values in the selected config.
The `mock` provider can run local examples without a real provider key.

For the full code and data-flow map, see
[docs/architecture.md](../docs/architecture.md).
