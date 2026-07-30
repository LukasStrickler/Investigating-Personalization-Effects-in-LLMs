# Investigating Personalization Effects in LLMs

Do LLMs infer user identity from multi-turn conversation history, and does that
change job and college-major advice?

This repo holds the study's data, experiment code, committed model outputs, and
notebooks that reproduce the paper's main Results figures. Start with the
research questions below; setup and re-runs come later.

University of Mannheim research project. License: [MIT](LICENSE).

## Research questions

| RQ | Question | What we measure |
| --- | --- | --- |
| **RQ1** | Can models infer gender and region from conversation history? | Direct probing (ask the model outright) and linear probes on hidden-state activations |
| **RQ2** | Do those inferences change recommendations? | Behavioral audit of job recommendations and college-major recommendations |

### Data and pipeline (brief)

We build about 3,869 synthetic multi-turn conversation histories from **26
personas** (16 full gender × region cells, plus gender-only and region-only
partials; see [`docs/combination_analysis.md`](docs/combination_analysis.md)).
Identity cues include names, music, hobbies, and movies. We also keep 50
[WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) histories as a
real-world check.

Each **subject model** (the LLM under study) answers in free text. A separate
**LLM judge** (GPT-4o-mini via OpenRouter) maps answers
onto standard taxonomies: [ISCO-08](https://www.ilo.org/publications/international-standard-classification-occupations-2008-isco-08) job groups and
[ISCED-F 2013](https://uis.unesco.org/sites/default/files/documents/isced-fields-of-education-and-training-2013-en.pdf)
fields of study. Refusals and unclear answers get a reserved None-style label
(`__NONE__` in direct-probing CSVs).

Experiment runners write two CSV stages:

1. **Stage 1** - the subject model's free-text answer
2. **Stage 2** - the judge's structured label

Eval notebooks and [`finalresults.ipynb`](finalresults.ipynb) only read those
outputs (or committed `results_*`). Live runs go to gitignored `logs/`.

### Headline pattern

Gender and region are both easy to recover under direct probing. Gender shifts
job and major recommendations across models. Region mostly does not, even when
region is still easy to infer.

## Repository map

Click a name to open its README (or best guide). `run_*.py` writes CSVs;
`eval_*.ipynb` / [`finalresults.ipynb`](finalresults.ipynb) only read them.

- [`finalresults.ipynb`](finalresults.ipynb) - main Results figures
- [`experiments/`](experiments/README.md)
  - [`direct_probing/`](experiments/direct_probing/README.md) - RQ1 ask gender / region
  - [`internal_representation/`](experiments/internal_representation/README.md) - RQ1 hidden-state probes
  - [`behavioral_audit/`](experiments/behavioral_audit/README.md) - RQ2 job / major audit
  - [`judge_audit/`](experiments/judge_audit/README.md) — human-rater validation of the stage-2 judge
- `src/`
  - [`generate_backgrounds/`](src/generate_backgrounds/README.md) - synthetic personas
  - [`real_conversation_histories/`](src/real_conversation_histories/README.txt) - WildChat extraction
  - [`inference/`](docs/architecture.md) - shared client
- [`scripts/`](scripts/README.md)
  - [`slurm/`](docs/running-vllm-on-clusters.md) - cluster launchers
  - [`modal/`](scripts/modal/README.md) - Modal setup + deploy
  - [`estimate_cost.py`](scripts/estimate_cost.py)
- [`config/`](config/README.md) · [`docs/`](docs/README.md) · [`examples/`](examples/README.md) · [`tests/`](tests/)

## Methods by research question

### RQ1: identity inference

**Direct probing** asks the subject model to state the user's gender and region
from the history; the judge maps the free-text guess. Committed judgments for
analysis:
[`stage2/postprocessed/`](experiments/direct_probing/results_direct_probing/stage2/postprocessed/).
How refusals become `__NONE__`, plus per-run notebooks:
[`experiments/direct_probing/README.md`](experiments/direct_probing/README.md).

**Internal probes** train simple linear classifiers on the model's hidden
states (no fine-tuning). Report layer-sweep figures use
[`results_modal_gemma4_31b_sweep_v4/`](experiments/internal_representation/results_modal_gemma4_31b_sweep_v4/).
Other committed Modal runs (Gemma 4 31B / E2B, Ministral 3 8B, ablation):
[`experiments/internal_representation/README.md`](experiments/internal_representation/README.md).

### RQ2: behavioral audit

The same histories end with a job or college-major question. Stage 1 is subject
advice; stage 2 is the judge mapping into the taxonomies under
`indicator_hierarchy/`. Canonical committed tree (prefer this over any top-level
partial copies):
[`results_behavioral_audit/`](experiments/behavioral_audit/results_behavioral_audit/).

Subjects in the committed runs: Gemma 4 31B, DeepSeek V4 Flash, Grok 4.3, and
GLM-5.2 (via OpenRouter); Gemma 4 E2B and Ministral 3 8B (via Modal); plus
no-persona baselines. Layout, WildChat, and eval notebooks:
[`experiments/behavioral_audit/README.md`](experiments/behavioral_audit/README.md).

[`finalresults.ipynb`](finalresults.ipynb) is the entry point for the main Results
plots across RQ1 and RQ2. Appendix / baseline / ablation plots stay in the
per-experiment `eval_*.ipynb` notebooks.

## Getting started

### Main path: rebuild Results figures

No API key and no GPU. This is enough to inspect the study outputs:

```bash
uv sync
uv run jupyter lab finalresults.ipynb
```

[`finalresults.ipynb`](finalresults.ipynb) reads the committed CSVs linked under
[Methods](#methods-by-research-question) and regenerates the main Results plots
for direct probing, internal probing, the behavioral audit, and WildChat. It
does not call models. Appendix, baseline, and ablation plots live in the
per-experiment `eval_*.ipynb` notebooks.

Merged significance tables live in
[`experiments/behavioral_audit/README.md`](experiments/behavioral_audit/README.md).

### Setup (when you need more than figures)

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) ([project notes](docs/uv.md))

```bash
uv sync                        # finalresults + runners + eval notebooks
uv sync --extra internal-rep   # activation probes (torch / transformers)
uv sync --extra modal          # Modal deploy SDK
uv sync --extra vllm           # local/cluster vLLM server package
uv sync --extra dev            # pytest, ruff, mypy
```

| Extra | Needed for |
| --- | --- |
| (core) | `finalresults.ipynb`, OpenRouter runners, eval notebooks |
| `internal-rep` | Re-running activation probes |
| `modal` / `vllm` | Optional GPU serving |
| `dev` | Tests and linting |

### Optional: re-run inference

Only if you want **new** model outputs. Costly; Modal/Slurm need GPUs. Not
required for `finalresults.ipynb`.

```bash
cp .env.example .env                                    # set OPENROUTER_API_KEY
cp config/inference.example.yaml config/inference.yaml  # or .modal. / .vllm. examples
```

Config aliases: [`config/README.md`](config/README.md). Then pick a runner:

| Experiment | Start here |
| --- | --- |
| RQ1 direct probing | [`experiments/direct_probing/README.md`](experiments/direct_probing/README.md) |
| RQ1 internal probes | [`experiments/internal_representation/README.md`](experiments/internal_representation/README.md) |
| RQ2 behavioral audit | [`experiments/behavioral_audit/README.md`](experiments/behavioral_audit/README.md) |

Compute backends:

- OpenRouter (API): copy `inference.example.yaml`, set the key, run the OpenRouter
  `run_*.py` scripts in the experiment folder
- Modal (GPU): [`scripts/modal/README.md`](scripts/modal/README.md)
- Cluster vLLM: [`docs/running-vllm-on-clusters.md`](docs/running-vllm-on-clusters.md)
- Cost ballpark: [`scripts/estimate_cost.py`](scripts/estimate_cost.py)

Persona / WildChat regeneration is also optional; see the `src/` READMEs in the
map above.

## Development

```bash
uv sync --extra dev
uv run pytest tests -q
uv run mypy src --ignore-missing-imports
uv run ruff check .
```

## Further reading

| Doc | Covers |
| --- | --- |
| [`docs/combination_analysis.md`](docs/combination_analysis.md) | Persona combination space (full vs partial) |
| [`docs/experiments-usage.md`](docs/experiments-usage.md) | Inference harness API (matrices, resume) |
| [`docs/uv.md`](docs/uv.md) | `uv` workflow and extras |
| [`examples/`](examples/README.md) | Small library notebooks |
| [`docs/README.md`](docs/README.md) | Full documentation index |
