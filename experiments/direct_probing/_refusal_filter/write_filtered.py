"""Write filtered stage-2 CSV copies with refusals dropped — PER MODEL.

prompt_ids overlap across models (same probing subjects), so refusals must be applied
per model, not globally. Reads verdicts_by_model.json: {model: {prompt_id: {verdict,
refused_gender, refused_background}}}. A row is dropped if its model's verdict is
"refusal" (model refused EITHER gender OR background).

Writes *.judgments.filtered.csv next to each original (originals untouched) and a
refusal_summary.csv with per-model counts. e2b is included too.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

BY_MODEL = json.loads((HERE / "verdicts_by_model.json").read_text())

MM = "logs/judges/direct-probing/direct-probing-combined-direct_multimodel001"
STAGE2 = {
    "gemma-4-31b_paid": REPO / "logs/judges/direct-probing/direct-probing-combined-direct_complete002-stage2.judgments.csv",
    "deepseek-v4-flash_paid": REPO / f"{MM}-deepseek-v4-flash_paid-stage2.judgments.csv",
    "glm-5.2_paid": REPO / f"{MM}-glm-5.2_paid-stage2.judgments.csv",
    "grok-4.3_paid": REPO / f"{MM}-grok-4.3_paid-stage2.judgments.csv",
    "ministral-3-8b_modal": REPO / "experiments/direct_probing/results_direct_probing/direct-probing-combined-ministral3-8b-ministral-3-8b_modal-stage2.judgments.csv",
    "gemma-4-e2b_modal": REPO / "experiments/direct_probing/results_direct_probing/direct-probing-combined-e2b-gemma-4-e2b_modal-stage2.judgments.csv",
}

summary = []
for name, path in STAGE2.items():
    verdicts = BY_MODEL.get(name, {})
    refusal_ids = {pid for pid, vd in verdicts.items() if vd["verdict"] == "refusal"}
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    # keep non-refusal rows; rows lacking a verdict (e.g. status!=success) are kept as-is
    kept = [r for r in rows if r["prompt_id"] not in refusal_ids]
    dropped = len(rows) - len(kept)
    out = path.with_name(path.name.replace(".judgments.csv", ".judgments.filtered.csv"))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    summary.append({"model": name, "total_rows": len(rows), "verdicts": len(verdicts),
                    "dropped_refusals": dropped, "kept": len(kept)})
    print(f"{name:24s} rows={len(rows):4d} refusals_dropped={dropped:4d} kept={len(kept):4d} -> {out.name}")

with open(HERE / "refusal_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "total_rows", "verdicts", "dropped_refusals", "kept"])
    w.writeheader()
    w.writerows(summary)
print("\nsummary -> refusal_summary.csv")
