# direct_probing

Measures whether a subject model can **directly infer the user's gender and region**
from a conversation history. A subject model is asked to guess both attributes (stage 1),
then an LLM judge maps the free-text answer to a `gender - region` class (stage 2).

See the [experiments overview](../README.md) for how this fits the wider project, and
[`../../src/inference/`](../../src/inference/) for the inference layer the runners call.

## Flow

```
persona history → [run_direct_probing*.py]  → stage-1 subject answer
                → [judge]                    → stage-2 judgments.csv  (final_class = "gender - region")
                → [postprocess_none.ipynb]   → results_direct_probing/stage2/postprocessed/*.postprocessed.csv
                → [eval_direct_probing.ipynb]→ figures / accuracy tables
```

## Runners

| Script                             | Purpose                                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------------- |
| `run_direct_probing.py`            | Single-model run. (gemma-4-31b)                                                                     |
| `run_direct_probing_multimodel.py` | Several models in one run (each writes its own stage-2 CSV). (deepseek-v4-flash, grok-4.3, glm-5.2) |
| `run_direct_probing_modal.py`      | Modal-hosted subject models (ministral, e2b).                                                       |
| `run_direct_probing_wildchat.py`   | Runs on real WildChat histories (gender only).                                                      |
| `show_progress_direct_probing.py`  | Live progress of an in-flight run.                                                                  |

## Evaluation & postprocessing

- **`postprocess_none.ipynb`** — the subject model sometimes declines to state an
  attribute, but the judge is forced to emit a class anyway. A per-axis refusal pass
  (`_refusal_filter/`) marks each declined axis, and this notebook relabels it as
  `__NONE__` in `final_class`, writing
  `results_direct_probing/stage2/postprocessed/*.judgments.postprocessed.csv`.
  Run this **before** the eval notebook.
- **`eval_direct_probing.ipynb`** — main evaluation (accuracy, confusion matrices,
  per-axis NONE rate). Reads the postprocessed CSVs.
- `eval_direct_probing_gemma50.ipynb` — gemma-4-31b on a 50-row no-refusal subset.
- `eval_direct_probing_wildchat.ipynb` — evaluation on real WildChat histories.

## Directories

- `results_direct_probing/stage1/` — raw stage-1 subject responses (per run).
- `results_direct_probing/stage2/` — stage-2 judgment CSVs; `masked_csv/` holds the
  per-axis `REFUSED` copies; `postprocessed/` holds the `__NONE__`-relabelled outputs.
- `_refusal_filter/` — refusal verdicts (`verdicts_by_model.json`), dumped stage-1
  responses, and the scripts that build the masked CSVs. Provenance for the NONE values.
