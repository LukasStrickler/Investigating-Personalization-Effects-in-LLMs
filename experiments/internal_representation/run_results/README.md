# Gender Probing Experiment — Stereotyped vs Neutral Indicator Ablation

## What this experiment does

A logistic-regression **linear probe** is trained at every transformer layer on the
frozen hidden states of SmolLM2-360M-Instruct to classify whether a conversation
history belongs to a **Female** or **Male** persona.

The same 400 histories are presented in six different text forms that systematically
isolate how much gender information each indicator dimension carries:

| Condition | Movie indicator | Hobby indicator |
|---|---|---|
| **FULL** | present (e.g. `"action"`) | present (e.g. `"yoga"`) |
| **HOBBY_ONLY** | replaced with `<Movie>` | present |
| **MOVIE_ONLY** | present | replaced with `<Hobby>` |
| **NEUTRAL_BOTH** | replaced with neutral word (e.g. `"documentary"`) | replaced with neutral word (e.g. `"photography"`) |
| **NEUTRAL_HOBBY** | present | replaced with neutral word |
| **NEUTRAL_MOVIE** | replaced with neutral word | present |

Neutral words are drawn uniformly at random from a gender-blind pool, assigned
independently of the history's true gender (seed=99).

---

## Setup

- **Model**: SmolLM2-360M-Instruct (frozen weights, 32 transformer layers, hidden dim **960**)
- **Histories**: 400 conversations (200 Female + 200 Male), seed=42
- **Context mode**: `gender-turn-only` — each history is one user message:
  > *"I really enjoy watching `<movie>` movies, and outside of that I spend a lot
  > of time on `<hobby>`. As someone with my background, how should I structure
  > my daily routine?"*
- **Hidden states**: captured at the last user-turn token, at all 33 positions
  (embedding layer + 32 transformer blocks)
- **Fresh probe**: logistic regression, 5-fold stratified CV, retrained from scratch
  per condition; scaler fit per fold to prevent test-set leakage
- **Frozen FULL probe**: trained once on FULL at the best layer, then applied with
  locked weights to all other conditions for the P(Male) plots

---

## Files

```
run_results/
├── README.md
├── run.log                          execution log with per-layer accuracy
├── data/
│   ├── dataset_personas.json        400 conversation histories + gender labels
│   ├── hs_all.npz                   hidden states — FULL/HOBBY/MOVIE × 33 layers
│   ├── hs_neutral.npz               hidden states — NB/NH/NM × 33 layers
│   ├── phrase_stats.csv             per-indicator P(Male) in each stereotyped condition
│   └── neutral_word_stats.csv       per-neutral-word P(Male) for Female and Male histories
└── plots/
    ├── probe_accuracy_comparison.png     stereotyped fresh-probe accuracy per layer
    ├── p_male_distribution.png           stereotyped P(Male) strip plots (frozen probe)
    ├── indicator_probability_by_run.png  per-word dots across conditions
    ├── word_presence_comparison.png      before/after arrows
    ├── neutral_probe_accuracy.png        stereotyped vs neutral accuracy side-by-side
    ├── neutral_p_male_distribution.png   2×3 grid all 6 conditions (frozen probe)
    └── neutral_word_p_male.png           per-neutral-word Female vs Male P(Male)
```

---

## Quantitative results

| Condition | Fresh probe acc | Female P(Male) | Male P(Male) |
|---|---|---|---|
| FULL (both stereotyped) | **1.000** | 0.000 | 1.000 |
| HOBBY_ONLY | **1.000** | 0.659 | 0.999 |
| MOVIE_ONLY | **1.000** | 0.960 | 1.000 |
| NEUTRAL_BOTH | **0.483** | 0.577 | 0.601 |
| NEUTRAL_HOBBY | **1.000** | 0.012 | 0.966 |
| NEUTRAL_MOVIE | **1.000** | 0.210 | 0.944 |
| Shuffled control | 0.463 | — | — |

*Best layer: 1 (one attention pass after embedding). P(Male) values are from the frozen FULL probe.*

**Neutral word P(Male) in NEUTRAL_BOTH** (sorted by Female P(Male)):

| Word | Dimension | Female P(Male) | Male P(Male) | Gap | Interpretation |
|---|---|---|---|---|---|
| independent | Movie | 0.079 | 0.067 | 0.012 | LLM codes "independent films" as Female-coded |
| budgeting | Hobby | 0.128 | 0.184 | 0.056 | LLM codes "budgeting" as Female-coded |
| traveling | Hobby | 0.314 | 0.210 | 0.104 | Near-neutral |
| international | Movie | 0.411 | 0.364 | 0.047 | Near-neutral |
| adventure | Movie | 0.439 | 0.565 | 0.126 | Near-neutral |
| cooking | Hobby | 0.480 | 0.450 | 0.030 | Neutral ✓ |
| journaling | Hobby | 0.543 | 0.451 | 0.092 | Neutral ✓ |
| cycling | Hobby | 0.587 | 0.663 | 0.076 | Near-neutral |
| cult | Movie | 0.597 | 0.616 | 0.019 | Neutral ✓ |
| historical | Movie | 0.619 | 0.589 | 0.030 | Neutral ✓ |
| crime | Movie | 0.625 | 0.630 | 0.005 | Neutral ✓ |
| photography | Hobby | 0.634 | 0.636 | 0.002 | Neutral ✓ |
| swimming | Hobby | 0.658 | 0.718 | 0.060 | Near-neutral |
| hiking | Hobby | 0.661 | 0.777 | 0.116 | Near-neutral |
| documentary | Movie | 0.695 | 0.692 | 0.003 | Neutral ✓ |
| biographical | Movie | 0.726 | 0.728 | 0.002 | Neutral ✓ |
| mystery | Movie | 0.778 | 0.759 | 0.019 | Near-neutral |
| writing | Hobby | 0.809 | 0.845 | 0.036 | Near-neutral |
| baking | Hobby | 0.931 | 0.913 | 0.018 | LLM codes "baking" as Male-coded (counterintuitive) |
| foreign | Movie | 0.933 | 0.953 | 0.020 | LLM codes "foreign films" as Male-coded |

---

## Key findings

**1. NEUTRAL_BOTH accuracy = 0.483 — at chance.**
Replacing both stereotyped indicators with neutral words drops probe accuracy to below
random guessing. No learnable gender signal remains; the probe is entirely vocabulary
detection, not gender understanding.

**2. One stereotyped word is enough — NEUTRAL_HOBBY and NEUTRAL_MOVIE both reach 1.000.**
Keeping a single gender-coded indicator (movie or hobby) while neutralising the other
is sufficient for the probe to classify perfectly. The signal is redundant across both
indicator types.

**3. For neutral words, Female and Male histories get the same P(Male).**
Truly neutral words (photography, crime, documentary, cooking, etc.) yield essentially
identical probe outputs for Female and Male histories (gap < 0.01–0.03). The word
determines the output; the persona's actual gender does not.

**4. Some "neutral" words carry unexpected LLM-internal stereotypes.**
These biases apply equally to both Female and Male histories — they are word-level
associations, not persona-level information:
- *independent movies* → both groups score ~0.07 (Female-coded in LLM)
- *foreign movies* → both groups score ~0.94 (Male-coded in LLM)
- *baking* → both groups score ~0.92 (unexpectedly Male-coded in LLM)
- *budgeting* → both groups score ~0.16 (Female-coded in LLM)

**5. Layer 0 = 0.500; Layer 1 = 1.000.**
The token-embedding layer has no information (last token cannot "see" the indicator
words without attention). A single attention pass is sufficient to broadcast the
indicator word's identity to the last position, enabling perfect classification.

---

## Plots

### `probe_accuracy_comparison.png`

Fresh probe accuracy at each layer for FULL, HOBBY_ONLY, MOVIE_ONLY, and the
shuffled-label control.  All three real conditions reach 100 % — either indicator
alone is sufficient for any linear classifier trained at layer ≥ 1.

---

### `p_male_distribution.png`

Strip plots showing P(Male) from the frozen FULL probe for every history under
each stereotyped condition.  Perfect separation (blue at 0, orange at 1) in FULL
collapses when one dimension is removed, revealing the probe's geometric reliance
on each indicator type.

---

### `indicator_probability_by_run.png`

Per-word view.  Three markers (● FULL, ■ HOBBY_ONLY, ▲ MOVIE_ONLY) show the
frozen-probe mean P(Male) for each indicator across conditions.  Saturated colour =
that indicator type is present; grey = replaced with placeholder.

---

### `word_presence_comparison.png`

Arrow charts: filled circle = FULL baseline, hollow diamond = after one dimension
is removed.  Arrow direction shows whether removing movie or hobby pushes a word's
histories toward Male (the unmarked default).

---

### `neutral_probe_accuracy.png`

Side-by-side accuracy plot: stereotyped (left) vs neutral (right) conditions.
NEUTRAL_BOTH collapses to chance immediately; NEUTRAL_HOBBY and NEUTRAL_MOVIE
stay at 100 % because one stereotyped indicator survives.

---

### `neutral_p_male_distribution.png`

2 × 3 grid of P(Male) strip plots covering all six conditions.  The bottom-left
panel (NEUTRAL_BOTH) shows both groups collapsing to ~0.58 — no separation,
no information.

---

### `neutral_word_p_male.png`

Per-neutral-word scatterplot: blue circle = mean P(Male) for Female histories
assigned that word, orange diamond = mean P(Male) for Male histories assigned the
same word.  Points at the same horizontal position confirm the word drives the
probe, not the persona's gender.  A large vertical gap would indicate residual
gender signal — none is found for truly neutral words.

---

## How the two probe types work — and why they measure different things

### Step 0 — The LLM is always frozen

SmolLM2-360M-Instruct is a pretrained language model.  Its weights are never
updated in this experiment.  We only ever *read* its internal representations.

For each conversation history, we run one forward pass through the model and record
the hidden state vector (960 numbers) at the last token of the user turn, at every
transformer layer.  Different texts produce different vectors; the model never changes.

---

### Probe type 1 — Fresh probe (one per condition, trained and evaluated independently)

For each condition (FULL, HOBBY_ONLY, MOVIE_ONLY, …) we do the following completely
separately:

1. Pass all 400 texts for that condition through the frozen LLM → 400 hidden
   state vectors at each layer.
2. Train a logistic regression classifier on those 400 vectors using 5-fold
   stratified cross-validation.  The StandardScaler is fit on the training fold
   only (not on all data) to avoid test-set leakage.
3. Report average test accuracy.

The fresh probe answers: **"Is there enough gender information left in the hidden
state for any linear classifier to exploit?"**

Because the probe is retrained from scratch each time, it can find a new decision
boundary specific to whichever input was provided — even if that boundary points in
a completely different direction from what the FULL probe learned.

Result: all stereotyped conditions reach 100 %; NEUTRAL_BOTH falls to 0.483 (chance).

---

### Probe type 2 — Frozen FULL probe (trained once on FULL, weights locked)

One logistic regression is trained on the FULL condition hidden states at the best
layer.  Its weights (the decision boundary) are then **frozen** — the regression
coefficients are saved and never updated again.

That same frozen classifier is applied to the hidden states from all other
conditions.  No retraining occurs.

The frozen probe answers: **"When the input text changes, how far does the hidden
state shift relative to the boundary the model originally learned from full inputs?"**

This reveals the geometry of the representation space.  If the hidden state for a
Female history barely moves when the movie is removed, the Female signal was coming
from the hobby all along.  If it collapses to the Male side when the hobby is
removed, the hobby was the primary geometric driver.

---

### Concrete example — what "frozen" means step by step

```
Text:    "I enjoy watching drama movies ... yoga ..."       ← FULL text
                                      ↓
                        LLM forward pass (weights locked)
                                      ↓
Hidden state at layer 1:  [0.12, -0.83, 0.44, ...]        ← 960-dim vector
                                      ↓
            Full probe (decision boundary also locked)
                                      ↓
            P(Male) = 0.001   →  classified as Female ✓
```

Now remove the hobby (MOVIE_ONLY):

```
Text:    "I enjoy watching drama movies ... <Hobby> ..."   ← MOVIE_ONLY text
                                      ↓
                        LLM forward pass (same frozen LLM)
                                      ↓
Hidden state at layer 1:  [0.09, -0.11, 0.41, ...]        ← different vector!
                                      ↓
            Full probe (same frozen decision boundary)
                                      ↓
            P(Male) = 0.954   →  classified as Male ✗
```

The LLM produced a different hidden state because the input text changed.  The
probe did not retrain — it applied the exact same boundary.  Without the hobby
word, the hidden state moved close to the Male default region.

---

### Summary of results

| Probe type | FULL | HOBBY_ONLY acc / Female P(Male) | MOVIE_ONLY acc / Female P(Male) | NEUTRAL_BOTH |
|---|---|---|---|---|
| **Fresh** (retrained per condition) | 1.000 | 1.000 / — | 1.000 / — | **0.483** |
| **FULL-frozen** (P(Male) for Female histories) | 0.000 | — / 0.659 | — / 0.960 | — / 0.577 |

- Fresh probe: either indicator alone is enough (100 % both ways); both neutral → chance.
- Frozen probe: removing hobby collapses Female histories almost entirely to Male
  (0.960), while removing movie leaves them near the boundary (0.659).  The probe's
  decision boundary relies more on the hobby dimension to separate Female from Male.
