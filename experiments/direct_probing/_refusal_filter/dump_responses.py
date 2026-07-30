"""Extract (prompt_id, model response) pairs for the 5 kept direct-probing models.

Writes one JSON per model into ./ (alongside this script). Each JSON is a list of
{"prompt_id", "response", "final_class", "true_gender", "true_region"} — everything a
classifier needs to judge whether the model refused to predict gender/background.

Originals are only read, never modified.
"""
import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

MM_S1 = REPO / "experiments/direct_probing/results_direct_probing/direct-probing-combined-direct_multimodel001-stage1/20260714T175340.csv"

# model -> (stage2 csv, stage1 csv, response-column alias or None=first alias col)
MODELS = {
    "gemma-4-31b_paid": (
        REPO / "logs/judges/direct-probing/direct-probing-combined-direct_complete002-stage2.judgments.csv",
        next(iter(glob.glob(str(REPO / "experiments/direct_probing/results_direct_probing/direct-probing-combined-direct_complete002-stage1/*.csv")))),
        None,
    ),
    "deepseek-v4-flash_paid": (
        REPO / "logs/judges/direct-probing/direct-probing-combined-direct_multimodel001-deepseek-v4-flash_paid-stage2.judgments.csv",
        MM_S1, "deepseek-v4-flash_paid",
    ),
    "glm-5.2_paid": (
        REPO / "logs/judges/direct-probing/direct-probing-combined-direct_multimodel001-glm-5.2_paid-stage2.judgments.csv",
        MM_S1, "glm-5.2_paid",
    ),
    "grok-4.3_paid": (
        REPO / "logs/judges/direct-probing/direct-probing-combined-direct_multimodel001-grok-4.3_paid-stage2.judgments.csv",
        MM_S1, "grok-4.3_paid",
    ),
    "ministral-3-8b_modal": (
        REPO / "experiments/direct_probing/results_direct_probing/direct-probing-combined-ministral3-8b-ministral-3-8b_modal-stage2.judgments.csv",
        next(iter(glob.glob(str(REPO / "experiments/direct_probing/results_direct_probing/direct-probing-combined-ministral3-8b-stage1/*.csv")))),
        "ministral-3-8b_modal",
    ),
}


def response_map(csv_path, alias):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    aliases = [c for c in rows[0] if c not in ("prompt_id", "prompt", "prompt_metadata")]
    col = alias if alias in aliases else aliases[0]
    out = {}
    for r in rows:
        v = r[col] or ""
        try:
            o = json.loads(v)
            v = o.get("response", v) if isinstance(o, dict) else v
        except Exception:
            pass
        out[r["prompt_id"]] = v
    return out


for name, (s2, s1, alias) in MODELS.items():
    rmap = response_map(s1, alias)
    with open(s2) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "success"]
    items = []
    for r in rows:
        meta = json.loads(r["metadata"]) if r.get("metadata", "").strip() else {}
        items.append({
            "prompt_id": r["prompt_id"],
            "response": rmap.get(r["prompt_id"], ""),
            "final_class": r["final_class"],
            "true_gender": meta.get("true_gender"),
            "true_region": meta.get("true_region"),
        })
    outfile = OUT / f"{name}.responses.json"
    outfile.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"{name:24s} wrote {len(items):4d} -> {outfile.name}")
