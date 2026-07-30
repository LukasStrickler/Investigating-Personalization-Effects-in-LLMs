"""Write MASKED stage-2 CSV copies — per-axis refusal masking (no rows dropped).

prompt_ids overlap across models, so verdicts are applied per model from
verdicts_by_model.json: {model: {prompt_id: {verdict, refused_gender, refused_background}}}.

Instead of dropping refusal rows, every success row is kept but the REFUSED axis of
`final_class` is replaced with the sentinel "REFUSED":
    refused_gender      -> "REFUSED - <race>"
    refused_background  -> "<gender> - REFUSED"
    both                -> "REFUSED - REFUSED"
This lets the eval keep a committed axis toward its accuracy while excluding the
refused axis (the notebook treats "REFUSED" as "no prediction" for that axis).

Writes *.judgments.masked.csv next to each original (originals untouched) and a
refusal_summary.csv with per-model per-axis counts. e2b included.
"""
import csv
import json
from pathlib import Path

csv.field_size_limit(10 ** 7)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SENTINEL = "REFUSED"

BY_MODEL = json.loads((HERE / "verdicts_by_model.json").read_text())

MM = "logs/judges/direct-probing/direct-probing-combined-direct_multimodel001"
# ministral + e2b stage-2 CSVs live under results_direct_probing/stage2/
STAGE2_DIR = "experiments/direct_probing/results_direct_probing/stage2"
STAGE2 = {
    "gemma-4-31b_paid": REPO / "logs/judges/direct-probing/direct-probing-combined-direct_complete002-stage2.judgments.csv",
    "deepseek-v4-flash_paid": REPO / f"{MM}-deepseek-v4-flash_paid-stage2.judgments.csv",
    "glm-5.2_paid": REPO / f"{MM}-glm-5.2_paid-stage2.judgments.csv",
    "grok-4.3_paid": REPO / f"{MM}-grok-4.3_paid-stage2.judgments.csv",
    "ministral-3-8b_modal": REPO / f"{STAGE2_DIR}/direct-probing-combined-ministral3-8b-ministral-3-8b_modal-stage2.judgments.csv",
    "gemma-4-e2b_modal": REPO / f"{STAGE2_DIR}/direct-probing-combined-e2b-gemma-4-e2b_modal-stage2.judgments.csv",
}

# all masked copies are collected in one folder alongside the stage2 CSVs
MASKED_OUT = REPO / STAGE2_DIR / "masked_csv"
MASKED_OUT.mkdir(parents=True, exist_ok=True)


def mask_final_class(final_class, refuse_g, refuse_b):
    parts = final_class.split(" - ", 1)
    gender = parts[0] if len(parts) == 2 else final_class
    race = parts[1] if len(parts) == 2 else ""
    if refuse_g:
        gender = SENTINEL
    if refuse_b:
        race = SENTINEL
    return f"{gender} - {race}"


summary = []
for name, path in STAGE2.items():
    verdicts = BY_MODEL.get(name, {})
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    masked_g = masked_b = 0
    for r in rows:
        v = verdicts.get(r["prompt_id"])
        if not v:
            continue
        rg, rb = v["refused_gender"], v["refused_background"]
        if rg or rb:
            r["final_class"] = mask_final_class(r["final_class"], rg, rb)
            masked_g += int(rg)
            masked_b += int(rb)

    out = MASKED_OUT / path.name.replace(".judgments.csv", ".judgments.masked.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    summary.append({"model": name, "rows": len(rows),
                    "masked_gender": masked_g, "masked_background": masked_b})
    print(f"{name:24s} rows={len(rows):4d} masked_gender={masked_g:4d} masked_background={masked_b:4d} -> {out.name}")

with open(HERE / "refusal_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "rows", "masked_gender", "masked_background"])
    w.writeheader()
    w.writerows(summary)
print("\nsummary -> refusal_summary.csv")
