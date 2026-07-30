# Test Report — 19 June 2026

## Tested environment

- Apple Silicon (`arm64`), 16 GB RAM, MPS acceleration
- Python 3.13 virtual environment in `.venv`
- PyTorch 2.12.1, Transformers 5.13.0.dev0, scikit-learn 1.9.0
- Local model: `models/SmolLM2-360M-Instruct` (360M parameters)

## End-to-end run

```bash
.venv/bin/python main.py \
  --model models/SmolLM2-360M-Instruct \
  --attributes Gender \
  --samples 40 \
  --device-map mps \
  --dtype float16 \
  --max-seq-length 2048
```

- 80 histories: 40 Female, 40 Male
- Token lengths: 1,041–1,678; no truncation at 2,048
- 33 hidden-state layers, each `(80, 960)`, all values finite
- Best held-out layer: 16
- Held-out accuracy: 1.000 (16/16)
- Best shuffled-label control: 0.6875
- Selectivity gap: 0.3125
- Full extraction and probing runtime: 117.5 seconds
- Cached rerun runtime: 2.8 seconds

The result confirms that the complete software path works. It is not yet a
publication-grade estimate: the dataset is structured, the sample is small,
and selecting the best layer on the same held-out set can inflate accuracy.

## Word ablation

`analyze_words.py` completed against the saved layer-16 probe and created
`results/word_importance_22.csv` plus `plots/word_importance_22.png`.

The 200-history aggregate run selected six words per labelled group and
performed 276 history/word interventions. Strong associations included
`romantic`, `drama`, and `comedy` for Female-labelled histories and `science`,
`fiction`, and `war` for Male-labelled histories.

## Behavioral generation

The local model produced a parseable Gender guess for all 200 histories but
answered `Male` every time: 100/100 Male histories correct and 0/100 Female
histories correct (50% overall). This demonstrates that linear decodability does
not imply that the normal language-generation head uses the signal correctly.

## Gemma access

`google/gemma-3-1b-it` returned `401 GatedRepoError` because no authenticated
Hugging Face token is configured and its license has not been accepted for the
current account. This is an external access restriction, not a pipeline error.
The public SmolLM2 model is therefore the tested default.
