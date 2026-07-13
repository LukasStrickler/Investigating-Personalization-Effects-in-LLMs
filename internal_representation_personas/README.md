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

## Difference from `internal_representation`

The original `internal_representation` folder is the baseline implementation.
It creates synthetic programming conversations labelled by expertise:

```text
novice / intermediate / expert
```

It then asks whether expertise can be decoded from the model's hidden states.

This copied implementation keeps the same basic method but changes the data and
target:

| Baseline | Persona version |
|---|---|
| Synthetic programming conversations | Generated persona conversation histories |
| Expertise label | Gender label |
| Novice/intermediate/expert classifier | Male/Female classifier |
| Tests expertise representation | Tests persona-gender representation |

Both folders therefore produce the same kind of layer-wise probing graph, but
they investigate different information.

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
.venv/bin/python analyze_words.py --dataset-index 22
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
.venv/bin/python aggregate_word_analysis.py --batch-size 16
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

```bash
cd internal_representation_personas
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_model.py
```

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
.venv/bin/python main.py \
  --attributes Gender \
  --samples 40 \
  --device-map mps \
  --dtype float16 \
  --max-seq-length 2048
```

For CPU execution:

```bash
.venv/bin/python main.py \
  --attributes Gender \
  --samples 40 \
  --device-map cpu \
  --dtype float32
```

Omit `--samples` to use every history with a Gender label. Reuse matching cached
hidden states with `--skip-extraction`.

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
