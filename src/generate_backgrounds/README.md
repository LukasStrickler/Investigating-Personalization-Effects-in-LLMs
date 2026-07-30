# generate_backgrounds

Generates LLM conversation history **backgrounds** that encode persona information. These backgrounds are injected before experiment prompts to study how LLMs respond differently based on perceived user identity.

## Concept

A **background** is a two-message conversation history (user prompt + LLM response) that describes a person through one identity dimension. A **persona** is a combination of dimension values (e.g. _Social_Status=Elite_ + _Age=Young Adult_). For each persona, all possible indicator combinations across its dimensions are crossed to produce a set of complete multi-turn **conversation histories** — ready to be used as context in downstream LLM experiments.

```
Dimension value  ──► indicator combos  ──► LLM call  ──► BackgroundRecord
                                                               │
                 (all dimensions crossed per persona)          │
                                                               ▼
                                                      ConversationHistory
```

---

## Current quick run

Only for Gender and Region dimensions including partial personas.

```
uv run python -m generate_backgrounds \
  --config config/inference.yaml \
  --model-alias gemma-4-31b \
  --dimensions Gender Region \
  --include-partial \
  --assemble
```

## Directory structure

```

src/generate_backgrounds/
│
│ # Source modules
├── __init__.py                Public API exports
├── __main__.py                Entry point for `python -m generate_backgrounds`
├── cli.py                     argparse CLI — argument parsing + summary printing
├── pipeline.py                Three-phase async pipeline + all data classes
├── combination.py             CSV loading + indicator combination generation
├── rendering.py               Template loading + placeholder rendering
│
│ # Static data
├── dimension_templates.json   One prompt template per dimension
│
│ # Dimension value mappings (production data — add CSVs here)
├── dimension_value_mapping/
│ ├── gender.csv
│ ├── gender_name.csv
│ └── region.csv
│
│ # How the region-related attributes (names, artists) were sampled — see note below
├── attributes_region_sampling/
│ ├── names/   Perceived-origin name selection → gender_name.csv, top4_names_per_region.*
│ └── artists/ Regional artist cues → selected_artists.*
│
│ # Generated output (created at runtime — safe to delete and regenerate)
└── data/
    ├── backgrounds/
    │ ├── Gender/
    │ │ └── gender.jsonl One BackgroundRecord per indicator combo
    │ └── Region/
    │   └── region.jsonl
    └── personas/
      └── personas.jsonl One ConversationHistory per persona × indicator cross

```

---

## Region attribute sampling (`attributes_region_sampling/`)

This folder documents **how the region-related attribute values were derived**; it is
**not part of the `generate_backgrounds` runtime pipeline** and is not imported by any
module here. It is standalone, offline analysis kept for provenance and reproducibility.

- **`names/`** — selects personal names by their *perceived* ethnic/regional origin and
  gender, from a European name-perception survey (Ghekiere et al., 2025). `names_analysis.py`
  produces `top4_names_per_region.*` and `gender_name.csv`.
- **`artists/`** — manually collected regional artist lists used as cultural/regional cues;
  `artists_analysis.py` produces `selected_artists.*`.

The connection to this package is only through the **outputs**: the results feed into the
`dimension_value_mapping/` CSVs (e.g. `gender_name.csv`) that the pipeline actually consumes.
Each subfolder has its own `README.md` with full methodology and data attribution.

---

## Module guide

### `combination.py` — CSV → indicator combos

Reads a dimension CSV and generates every possible `IndicatorCombo`.

**CSV format** (`Dimension_value`, `Indicator_name`, `Indicator_value`):

```

Dimension_value,Indicator_name,Indicator_value
Elite,Household income,89082
Elite,Household savings,142458
Elite,Occupation,Chief executive officer
Elite,Occupation,Barrister

```

**Logic**: indicators that appear once per dimension value are _scalars_ (fixed); indicators that appear multiple times are _lists_ (varying). The Cartesian product of all list indicators is taken while scalars are held constant:

```

Elite × {CEO, Barrister} → 2 combos, each with income=89082, savings=142458

```

Each combo gets a deterministic `combination_id` (SHA-256 of its content) used for deduplication.

---

### `rendering.py` — templates → rendered prompts

Loads `dimension_templates.json` and substitutes `<IndicatorName>` placeholders:

```json
"Social_Status": "I am a <Occupation> with income <Household income> and savings <Household savings>."
```

```python
render_template(template, {"Occupation": "CEO", "Household income": "89082", ...})
# → "I am a CEO with income 89082 and savings 142458."
```

`discover_dimensions()` finds all dimensions that have **both** a template entry and a matching CSV file (by lowercase name convention: `Social_Status` → `social_status.csv`).

---

### `pipeline.py` — three-phase async orchestration

**Config** — `GenerationConfig`:

| Field                   | Default                    | Purpose                                 |
| ----------------------- | -------------------------- | --------------------------------------- |
| `inference_config_path` | required                   | Path to `inference.yaml`                |
| `model_alias`           | required                   | Model to use for generation             |
| `templates_path`        | `dimension_templates.json` | Prompt templates                        |
| `mapping_dir`           | `dimension_value_mapping/` | CSV data directory                      |
| `output_dir`            | `data/backgrounds/`        | Per-dimension background output         |
| `personas_dir`          | `data/personas/`           | Assembled persona history output        |
| `concurrency`           | `8`                        | Max parallel LLM requests               |
| `system_prompt`         | `None`                     | Optional system prompt for all requests |

**Phase 1 — LLM generation** (`run_generation`):

- For each pending `IndicatorCombo`, renders the template and calls the LLM
- Saves each result as a `BackgroundRecord` to `data/backgrounds/<Dimension>/<timestamp>.jsonl`
- Skips combos whose `combination_id` already exists in prior output files (resume support)
- Uses `asyncio.Semaphore` for concurrency + `threading.Lock` for safe file writes

**Phase 2 — Persona enumeration**:

- Personas = Cartesian product of dimension values across all dimensions with data
- E.g. with Social_Status (7 values) × Age (3 values) → 21 personas

**Phase 3 — History assembly** (`assemble_personas`):

- For each persona, takes the Cartesian product of its per-dimension `BackgroundRecord` lists
- Each element → one `ConversationHistory` (messages from all dimensions concatenated in order)
- Saves to `data/personas/<timestamp>.jsonl`
- Skips histories whose `history_id` already exists (resume support)

---

### `cli.py` + `__main__.py` — command-line interface

Run from the project root:

```bash
uv run python -m generate_backgrounds \
  --config config/inference.yaml \
  --model-alias <alias>
```

| Flag                  | Default                    | Purpose                                                                 |
| --------------------- | -------------------------- | ----------------------------------------------------------------------- |
| `--config`            | required                   | Path to `inference.yaml`                                                |
| `--model-alias`       | required                   | Model alias from the inference config                                   |
| `--dimensions`        | all                        | Restrict to specific dimensions (e.g. `--dimensions Gender Region`)       |
| `--mapping-dir`       | `dimension_value_mapping/` | Override data directory (use for test data)                             |
| `--output-dir`        | `data/backgrounds/`        | Override background output path                                         |
| `--personas-dir`      | `data/personas/`           | Override persona output path                                            |
| `--concurrency`       | `1`                        | Max parallel LLM requests                                               |
| `--assemble`          | off                        | Run persona assembly (Phases 2+3) after background generation           |
| `--include-partial`   | off                        | Include partial personas (some dimensions set to `None`)                |
| `--persona DIM=VALUE` | none                       | Filter assembly to a specific persona; repeatable. Implies `--assemble` |
| `--system-prompt`     | none                       | System prompt for all generation requests                               |
| `-v` / `--verbose`    | off                        | Print detailed per-request logging                                      |

---

## Output format

**`data/backgrounds/<Dimension>/<timestamp>.jsonl`** — one record per LLM call:

```json
{
	"schema_version": 1,
	"dimension": "Social_Status",
	"dimension_value": "Elite",
	"combination_id": "f6bfe047...",
	"indicators": {
		"Household income": "89082",
		"Household savings": "142458",
		"Occupation": "Chief executive officer"
	},
	"model_alias": "gemma-3-4b",
	"messages": [
		{
			"role": "user",
			"content": "Please help me benchmark my socio-economic situation..."
		},
		{ "role": "assistant", "content": "Based on UK social structure..." }
	],
	"generated_at": "2026-04-13T14:35:00.000000Z"
}
```

**`data/personas/<timestamp>.jsonl`** — one record per complete conversation history:

```json
{
	"history_id": "a1b2c3...",
	"persona": { "Social_Status": "Elite", "Age": "Young Adult" },
	"combination_ids": { "Social_Status": "f6bfe047...", "Age": "3a9d..." },
	"messages": [
		{ "role": "user", "content": "Please help me benchmark..." },
		{ "role": "assistant", "content": "Based on UK social structure..." },
		{ "role": "user", "content": "I am currently a student..." },
		{ "role": "assistant", "content": "As a student, your next steps..." }
	],
	"generated_at": "2026-04-13T14:35:00.000000Z"
}
```

---

## Adding a new dimension

1. Add a CSV to `dimension_value_mapping/` named `<dimension_lowercase>.csv`
2. Add a matching entry to `dimension_templates.json` using `<IndicatorName>` placeholders
3. Run the pipeline — the new dimension is discovered automatically

---

## Resume behaviour

Runs can be interrupted and resumed freely. On each run the pipeline scans existing output files for `combination_id` (backgrounds) and `history_id` (personas) and skips anything already generated. To regenerate from scratch, delete `data/`.

---

## Quick start

```bash
# Test with mock provider on the real mapping data (no API key needed)
uv run python -m generate_backgrounds \
  --config config/inference.yaml \
  --model-alias mock-test

# Production run
uv run python -m generate_backgrounds \
  --config config/inference.yaml \
  --model-alias gemma-4-31b_paid
```
