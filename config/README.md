# Config Directory

YAML configs for the inference layer. Research code uses model aliases from these files.

- `inference.example.yaml` — OpenRouter + mock (notebooks, Stage 2 judge on cluster)
- `inference.vllm.example.yaml` — vLLM provider (Stage 1 on cluster or local GPU). See [docs/running-vllm-on-clusters.md](../docs/running-vllm-on-clusters.md).
- `inference.modal.example.yaml` — Modal-hosted subject model (Stage 1 on rented GPU). See [experiments/modal_gpu_poc/README.md](../experiments/modal_gpu_poc/README.md).
- `judge_smoke.yaml` — small config for `examples/llm_judge_example.ipynb`

```bash
cp config/inference.example.yaml config/inference.yaml
cp .env.example .env
```

For vLLM: `cp config/inference.vllm.example.yaml config/inference.yaml`

For Modal (self-hosted subject on rented GPU): `cp config/inference.modal.example.yaml config/inference.yaml`, then `export MODAL_BASE_URL=…` to the URL printed by `modal deploy experiments/modal_gpu_poc/modal_serve.py`. See [experiments/modal_gpu_poc/README.md](../experiments/modal_gpu_poc/README.md) for URL pattern, throughput tuning, and bug-fix notes vs PR #28.

See [docs/architecture.md](../docs/architecture.md) for the full data-flow map.
