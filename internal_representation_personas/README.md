# Internal Probing with Persona Conversation Histories

This project tests whether a persona's gender is represented in the hidden
states of a locally executed language model.

The tested default model is `HuggingFaceTB/SmolLM2-360M-Instruct`. It is public,
requires no Hugging Face login, has approximately 360 million parameters, and
uses roughly 700 MB of local storage.

## Research question

The current pipeline answers this question:

> After the model reads a persona's conversation history, can a simple linear
> classifier recover whether the persona was labelled Male or Female from the
> model's internal representation?

It does **not** directly answer either of these stronger questions:

- Which words causally made the LLM generate a particular answer?
- Did the LLM actually use its internal gender representation when producing
  that answer?

Those questions require additional causal generation experiments. The included
word-ablation analysis only measures how removing words changes the saved
gender probe's score.

## Label and model input

Every row in `personas.jsonl` provides:

```text
Ground-truth label: persona["Gender"] → "Male" or "Female"
Model context:      messages           → complete conversation history
```

The conversation history contains alternating user and assistant messages. The
pipeline excludes histories without a Gender label by default. Race can be
probed as an additional target using `--attributes Gender Race`, but it is not
enabled by default.

## What happens during a run?

1. `dataset.py` loads the persona JSONL file and converts each message history
   into model input text.
2. The frozen local LLM processes every history. No weights are fine-tuned and
   no gradients are used.
3. `extraction.py` records the hidden state at the end of the final user message
   for every transformer layer. Because causal attention cannot look forward,
   this state does not contain information from the assistant reply that follows
   that user message in the stored history.
4. `probing.py` trains one logistic-regression classifier per layer to predict
   Male versus Female from that layer's hidden state.
5. A control classifier is trained with shuffled Gender labels. This estimates
   how much accuracy could be obtained from noise or accidental fitting.
6. The best probe is saved, and predictions are exported for held-out histories.

If the real probe clearly outperforms the shuffled-label control, Gender is
linearly decodable from that layer. This means the information is present in an
accessible form; it does not by itself prove that the model uses it to generate
its answer.

## Understanding the layer graph

`plots/probe_accuracy_Gender_logistic.png` shows:

- **X-axis:** transformer layer, from the input embedding to the final layer.
- **Y-axis:** Gender classification accuracy on held-out histories.
- **Blue line:** the real Male/Female probe.
- **Red dashed line:** the probe trained with shuffled labels.

If the blue line rises above the red line in middle or later layers, the model's
representation increasingly separates the two labelled persona groups. The
graph does not show individual words, attention branches, or generated answers.

`plots/selectivity_gap_Gender.png` shows the blue-minus-red accuracy difference
for every layer. Larger positive values indicate a clearer Gender signal.

## Individual histories and questions

`results/test_predictions_gender.csv` contains only held-out examples and
includes:

- the final user question;
- the true Male/Female label;
- the best probe's predicted label;
- prediction confidence and class probabilities;
- the dataset index and history ID.

This makes it possible to inspect which histories the probe classifies correctly
or incorrectly. It still describes the probe's decision, not an answer generated
by the LLM.

## Word ablation

`analyze_words.py` removes words from the user messages one at a time, reruns the
local LLM, and measures how much the best Gender probe's probability changes.

```bash
uv run python analyze_words.py --dataset-index 22
```

Outputs:

- `results/word_importance_22.csv`
- `plots/word_importance_22.png`

A positive importance means that removing the word reduced the probability of
the originally predicted Gender class. A negative importance means that removing
it strengthened the prediction. This is a perturbation analysis of the linear
probe, not a literal reconstruction of the LLM's reasoning.

### Aggregate analysis over a complete run

`aggregate_word_analysis.py` reads all 25 exact Gender indicator phrases from
`dimension_value_mapping/gender.csv`, adds ten fixed-template control words,
and removes each target from every history in which it occurs. Multiword values
are treated as complete phrases. The change in the saved probe's `P(Male)` is
aggregated across histories.

```bash
uv run python aggregate_word_analysis.py --batch-size 16
```

Outputs include `gender_indicator_presence.csv`,
`gender_phrase_ablation_summary.csv`, `gender_phrase_ablation_details.csv`,
`gender_indicator_label_prevalence.png`, and
`gender_phrase_ablation_effect.png`.

## Reproducible isolated final run

`run_final_200.sh` reproduces the consolidated 100 Female / 100 Male run in
`final_run_200/`. It uses `--context-mode gender-turn-only`, so Race turns,
gender-coded names, and stored replies cannot confound the Gender probe. See
`final_run_200/README.md` for the complete methods, interpretation, and results.

## Installation and model download

Dependencies for this subsystem live in the repo's `pyproject.toml` under the
`internal-rep` extra (torch, transformers, scikit-learn, huggingface-hub, joblib).
From the repo root:

```bash
uv sync --extra internal-rep
uv run python internal_representation_personas/download_model.py
```

(The legacy `requirements.txt` in this folder is superseded by the `internal-rep`
extra and kept only for reference.)

The model is downloaded to `models/SmolLM2-360M-Instruct`. Models and generated
artifacts are ignored by Git.

Google's larger `google/gemma-3-1b-it` can be used as an alternative. Its Gemma
license must first be accepted on Hugging Face, followed by authentication:

```bash
hf auth login
python download_model.py \
  --model google/gemma-3-1b-it \
  --output models/gemma-3-1b-it
```

## Running the probe

A tested balanced run with 40 histories per Gender class:

```bash
uv run python main.py \
  --attributes Gender \
  --samples 40 \
  --device-map mps \
  --dtype float16 \
  --max-seq-length 2048
```

For CPU execution:

```bash
uv run python main.py \
  --attributes Gender \
  --samples 40 \
  --device-map cpu \
  --dtype float32
```

Omit `--samples` to use every history with a Gender label. Reuse matching cached
hidden states with `--skip-extraction`.

## Running on Modal (cloud GPU)

Large open-weights models do not fit in laptop memory — `google/gemma-4-31B-it`
is ~61 GB in bf16, and the `gemma-4-26B-A4B` MoE keeps all ~25 B params resident
(~50 GB) despite its ~3.8 B active. `modal_run.py` runs the **same** pipeline
(`run_pipeline` in `main.py`) on a Modal GPU container and downloads the
`results/` + `plots/` artifacts back to your machine. Model weights are cached in
the shared `pers-hf-cache` Modal volume across runs.

Gated Gemma repos need the `huggingface-token` Modal secret. Sync it once with the
existing helper (accept the model licence on Hugging Face first):

```bash
uv run python experiments/modal_gpu_poc/setup_modal_hf.py \
  --model-id google/gemma-4-31B-it
```

**Smoke test first** — cheap model on a cheap GPU to confirm the wiring:

```bash
MODAL_IR_GPU=L4 modal run internal_representation_personas/modal_run.py \
  --model google/gemma-4-E2B-it --samples 8
```

**Full run** — the 31 B dense model needs an 80 GB GPU. `--samples 0` (the
default) uses every labelled history (~3773); pass a positive number for a
per-group subsample. Use `--out` to keep runs separated by model. Use
`--detach` so a multi-minute run survives a laptop sleep / network blip:

```bash
MODAL_IR_GPU=A100-80GB modal run --detach internal_representation_personas/modal_run.py \
  --model google/gemma-4-31B-it --samples 0 --dtype bfloat16 \
  --out results_modal_gemma4_31b
```

Artifacts land in `internal_representation_personas/results_modal/{results,plots}`
(override with `--out`), so local `results/` runs are never clobbered. The large
`hidden_states_personas.npz` intermediate is not downloaded by default; set
`MODAL_IR_KEEP_HIDDEN=1` to include it.

Results are **also** persisted to the Modal volume `pers-ir-results` under
`<out>/`, so a `--detach` run keeps its output even if the local client
disconnects before the download. Fetch them any time with:

```bash
modal volume get pers-ir-results results_modal_gemma4_31b/ \
  internal_representation_personas/results_modal_gemma4_31b
```

GPU picking (env `MODAL_IR_GPU`) by model size (bf16 weights + room for per-layer
hidden-state extraction):

| Model | Weights | GPU |
|---|---|---|
| `gemma-4-E2B-it` / `gemma-4-E4B-it` | ~10–16 GB | `L4`, `A10G` |
| `gemma-4-12B-it` | ~24 GB | `L40S`, `A10G`, `A100-80GB` |
| `gemma-4-26B-A4B-it` (MoE) | ~50 GB | `A100-80GB` |
| `gemma-4-31B-it` (dense) | ~61 GB | `A100-80GB`, `H100` |

## Outputs

- `data/dataset_personas.json`: normalized histories used by the run
- `results/hidden_states_personas.npz`: hidden states for all extracted layers
- `results/probe_gender.json`: real and control metrics for every layer
- `results/best_probe_gender.joblib`: fitted best-layer probe
- `results/test_predictions_gender.csv`: held-out per-history predictions
- `results/summary_personas.json`: compact result summary
- `plots/probe_accuracy_Gender_logistic.png`: accuracy by layer
- `plots/selectivity_gap_Gender.png`: real-minus-control gap by layer
- `plots/word_importance_<index>.png`: optional word-ablation result

The completed local validation is documented in `TEST_REPORT.md`.
