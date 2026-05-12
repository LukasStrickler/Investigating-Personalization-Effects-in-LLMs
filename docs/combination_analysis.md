# Combination Analysis

## Current Status (Gender + Race, full personas only)

|                                   | Count                      |
| --------------------------------- | -------------------------- |
| **Gender backgrounds**            | 77 (Male: 35, Female: 42)  |
| **Race backgrounds**              | 132 (11 regions × 12 each) |
| **Total backgrounds (LLM calls)** | 209                        |
| **Personas**                      | 22                         |
| **Conversation histories**        | 5,082                      |

### Gender dimension breakdown

- Male: 7 Hobbies × 5 Movies = 35 combos
- Female: 7 Hobbies × 6 Movies = 42 combos

### Race dimension breakdown

- 11 regions × (4 Names × 3 Artists) = 132 combos
- After gender filtering: 2 Names × 3 Artists = 6 combos per persona

### Histories per persona

- Male persona: 35 × 6 = 210 histories
- Female persona: 42 × 6 = 252 histories
- Total: 11 × 210 + 11 × 252 = 2,310 + 2,772 = **5,082**

---

## Impact of adding an Indicator_value

Each new value multiplies through all other indicators in that dimension and all combos in other dimensions.

| Addition                     | New combos       | Δ histories                  | New total |
| ---------------------------- | ---------------- | ---------------------------- | --------- |
| +1 Male Hobby (7→8)          | 8×5=40           | +11×5×6 = +330               | 5,412     |
| +1 Male Movie (5→6)          | 7×6=42           | +11×7×6 = +462               | 5,544     |
| +1 Female Hobby (7→8)        | 8×6=48           | +11×6×6 = +396               | 5,478     |
| +1 Female Movie (6→7)        | 7×7=49           | +11×7×6 = +462               | 5,544     |
| +1 Name (1 region, 1 gender) | 3×3=9            | +35 or 42 × 3 = +105 to +126 | ~5,190    |
| +1 Artist (1 region)         | 2×4=8 per gender | +(35+42)×2 = +154            | 5,236     |

---

## Impact of adding a new Indicator_name (with K values)

Adding a new indicator name introduces a multiplicative factor of K on top of existing combos.

| Addition                                 | Effect                   | Δ histories                       |
| ---------------------------------------- | ------------------------ | --------------------------------- |
| New indicator (K values) on Male         | Male combos: 35 → 35×K   | +11 × 35×(K−1) × 6 = +2,310×(K−1) |
| New indicator (K values) on Female       | Female combos: 42 → 42×K | +11 × 42×(K−1) × 6 = +2,772×(K−1) |
| New indicator (K values) on both         | Both multiply by K       | +5,082×(K−1)                      |
| New indicator (K values) per Race region | Region combos: 12 → 12×K | (K−1) × current total             |

Example: adding a new indicator with 3 values to both genders → +5,082 × 2 = +10,164 → **15,246** total.

---

## Growth behavior

| What you add                                                    | Growth type        | Why                                                                                                                      |
| --------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| +1 indicator value (e.g., +1 hobby)                             | **Linear**         | Adds a fixed number of histories each time. Going from 7→8→9 hobbies adds ~330 each step.                                |
| +1 indicator name (e.g., a new "Sport" indicator with K values) | **Multiplicative** | Multiplies all existing combos in that dimension by K. Two new indicators with 3 values each: total × 3 × 3 = total × 9. |
| +1 dimension                                                    | **Multiplicative** | Multiplies the entire history count by the number of combos in the new dimension.                                        |

The explosive growth comes from adding **new indicators or dimensions** (multiplicative), not from adding more values to existing indicators (linear).

---

## --include-partial

Partial personas (where one dimension is None) add a small number of single-dimension histories.

|                                      | Count                   |
| ------------------------------------ | ----------------------- |
| Gender-only personas (Male + Female) | 35 + 42 = 77 histories  |
| Race-only personas (11 regions)      | 11 × 12 = 132 histories |
| **Total partial histories added**    | **+209**                |
| **New grand total**                  | **5,291**               |

Partial personas add +4.1% on top of the full-persona total.
