# Linear Probing Pipeline

Detect whether an LLM internally encodes pre-labeled user attributes (expertise level, sentiment, political stance, etc.) by probing its hidden states with simple linear classifiers.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Dataset  │────▶│  2. Extract  │────▶│  3. Train    │────▶│  4. Control  │
│  Generation  │     │  Hidden      │     │  Linear      │     │  Evaluation  │
│              │     │  States      │     │  Probes      │     │  (Shuffled)  │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
   dataset.py         extraction.py         probing.py           probing.py
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (default: Gemma-4-E2B)
python main.py

# 3. Use a different model
python main.py --model mistralai/Mistral-7B-Instruct-v0.3

# 4. Run on CPU (slower but no GPU required)
python main.py --device-map cpu --dtype float32

# 5. More samples per class for better statistics
python main.py --samples 100
```

## Project Structure

| File               | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `config.py`        | Dataclass-based configuration for all pipeline components         |
| `dataset.py`       | Step 1 — Synthetic conversation dataset with labeled attributes   |
| `extraction.py`    | Step 2 — Forward pass through frozen LLM, hidden state extraction |
| `probing.py`       | Steps 3–4 — Linear probe training + selectivity control task      |
| `visualization.py` | Layer-accuracy plots, selectivity gaps, heatmaps                  |
| `main.py`          | Pipeline orchestrator with CLI arguments                          |

## Methodology

### Step 1 — Labeled Dataset

Synthetic multi-turn conversations are generated with known user attributes:

- **Expertise**: novice / intermediate / expert
- **Sentiment**: positive / negative / neutral

Each conversation is a \`(conversation_text, attribute_label)\` pair.

### Step 2 — Hidden State Extraction

The frozen LLM processes each conversation. Hidden states from the **residual stream** at every layer are captured at the **last user token** position (configurable to mean-pooling).

### Step 3 — Linear Probing

A **Logistic Regression** (and optionally **Linear SVM**) classifier is trained per layer on these hidden states. We deliberately keep the probe linear — a non-linear probe could learn to extract the attribute from raw text features, which would not prove the LLM itself encoded it.

### Step 4 — Selectivity Control

Labels are randomized and a control probe is trained on the same hidden states. If the real probe achieves significantly higher accuracy than the control, the LLM's representations meaningfully encode the attribute.

| Scenario        | Real Probe | Control Probe | Interpretation                   |
| --------------- | ---------- | ------------- | -------------------------------- |
| Strong encoding | ~90%       | ~33% (chance) | Attribute is clearly represented |
| Weak signal     | ~50%       | ~33%          | Some encoding, needs more data   |
| Artifact        | ~60%       | ~55%          | Probe is picking up noise        |

## Outputs

After running, you'll find:

- `results/summary.json` — per-attribute verdict (ENCODED / WEAK / NOT FOUND)
- `results/probe_<attribute>.json` — full metrics per layer
- `results/hidden_states.npz` — cached hidden states (reuse with `--skip-extraction`)
- `plots/probe_accuracy_<attribute>_logistic.png` — accuracy over layers
- `plots/selectivity_gap_<attribute>.png` — real−control accuracy gap
- `plots/heatmap_logistic.png` — all attributes × layers

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Hugging Face `transformers` ≥ 4.40
- scikit-learn, numpy, matplotlib
- GPU recommended (runs on CPU with `--device-map cpu --dtype float32`)
- Hugging Face token for gated models (`export HF_TOKEN=...`)

## Extending

**Add new attributes:** Edit `config.py` → `DataConfig.attributes` and add matching conversation templates in `dataset.py` → `TEMPLATES`.

**Use your own dataset:** Replace `dataset.py` with a loader that returns the same dict format:

```python
{
    "conversations": ["User: ...\nAssistant: ...", ...],
    "labels": {"attribute_name": ["value1", "value2", ...]}
}
```

**Probe a different model:** Pass `--model <hf_model_id>` on the command line.
