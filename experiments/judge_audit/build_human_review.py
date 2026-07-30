"""Build the two human-only judge-review sheets from the stratified 500 sample.

No LLM-as-judge cross-check: this produces exactly two CSVs for people to fill in
by hand, splitting the seed-stratified `judge_audit_sample_500.csv` into:

* `judge_audit_human_50.csv`  — the 50-row audit subset (in_audit_50 == True)
* `judge_audit_human_450.csv` — the remaining 450 rows (the stratified complement)

Each row shows, in reading order:
  input (probe_question) -> subject answer (subject_response) -> allowed labels
  (options) -> judge label (final_class) -> judge reasoning (raw_output),
then five blank reviewer columns (rev_1..rev_5) plus a corrected-label / consensus
/ notes block so we can score the judge's accuracy ourselves.

Usage:
    .venv/bin/python experiments/behavioral_audit/judge_audit/prepare_judge_audit_sample.py
    .venv/bin/python experiments/behavioral_audit/judge_audit/build_human_review.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(10**7)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "behavioral_audit"

SAMPLE_CSV = HERE / "judge_audit_sample_500.csv"
OUT_50 = HERE / "judge_audit_human_50.csv"
OUT_450 = HERE / "judge_audit_human_450.csv"

NONE_SENTINEL = "__NONE__"

JOBS_JSON = EXPERIMENTS_DIR / "jobs_classification.json"
COLLEGE_JSON = EXPERIMENTS_DIR / "college_classification.json"
PERSONAS_PATH = REPO_ROOT / "src" / "generate_backgrounds" / "data" / "personas" / "personas.jsonl"


def _load_option_sets() -> dict[str, list[str]]:
    """Return {question -> ordered allowed-label list}, matching the run configs."""
    jobs = json.loads(JOBS_JSON.read_text(encoding="utf-8"))
    college = json.loads(COLLEGE_JSON.read_text(encoding="utf-8"))

    q1 = [
        smg["name"]
        for mg in jobs["job_classification"]["major_groups"]
        for smg in mg["sub_major_groups"]
    ]
    q2 = [
        nf["name"]
        for bf in college["college_major_classification"]["broad_fields"]
        for nf in bf["narrow_fields"]
    ]

    races: set[str] = set()
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)["persona"]
            g, r = p.get("Gender"), p.get("Race")
            if g and r:
                races.add(r)
    direct = [f"{g} - {r}" for g in ("Male", "Female") for r in sorted(races)]

    return {
        "q1": q1 + [NONE_SENTINEL],
        "q2": q2 + [NONE_SENTINEL],
        "direct_probe": direct + [NONE_SENTINEL],
    }


# Trace-back / orientation columns.
CONTEXT_COLS = [
    "sample_rank", "judgment_id", "run_tag", "question",
    "subject_model_alias", "judge_alias",
]
# The core review payload, in reading order.
REVIEW_CORE_COLS = [
    "probe_question",     # INPUT — what the subject model was asked
    "subject_response",   # OUTPUT — the subject model's answer (what the judge classified)
    "options",            # the exact allowed labels (incl. __NONE__)
    "n_options",
    "final_class",        # JUDGE RESPONSE — the label the judge chose (__NONE__ if it declared none)
    "none_declared",      # True when the judge explicitly picked no class (final_class == __NONE__)
    "raw_output",         # JUDGE REASONING — the judge's explanation (if any)
]
# The five human verdicts + a place to record the corrected label / notes.
HUMAN_COLS = [
    "rev_1",              # TRUE = judge's label is the correct best label, FALSE = not
    "rev_2",
    "rev_3",
    "rev_4",
    "rev_5",
    "human_best_label",   # the corrected/consensus label when the judge was wrong
    "consensus",          # e.g. valid | invalid | 3/5 valid (split)
    "review_notes",
]
# Light provenance tail.
PROVENANCE_COLS = [
    "true_gender", "true_race", "history_id", "prompt_id", "subject_id",
    "stratum", "source_id",
]

FIELDNAMES = CONTEXT_COLS + REVIEW_CORE_COLS + HUMAN_COLS + PROVENANCE_COLS


def _build_row(r: dict, option_sets: dict[str, list[str]]) -> dict:
    q = r["question"]
    opts = option_sets.get(q)
    if opts is None:
        print(f"WARNING: no option set for question={q!r} (rank {r['sample_rank']})", file=sys.stderr)
        opts = [NONE_SENTINEL]
    # A judge that declared no class shows up as an empty final_class + none_declared;
    # surface it as the __NONE__ sentinel (a real member of the option set) so a
    # reviewer can agree/disagree instead of staring at a blank cell.
    none_declared = str(r.get("none_declared", "")).strip().lower() in ("true", "1", "yes")
    judge_label = NONE_SENTINEL if none_declared else r["final_class"]

    out = {c: r.get(c, "") for c in CONTEXT_COLS}
    out["probe_question"] = r["probe_question"]
    out["subject_response"] = r["subject_response"]
    out["options"] = " | ".join(opts)
    out["n_options"] = str(len(opts))
    out["final_class"] = judge_label
    out["none_declared"] = "True" if none_declared else "False"
    out["raw_output"] = r["raw_output"]
    for c in HUMAN_COLS:
        out[c] = ""
    for c in PROVENANCE_COLS:
        out[c] = r.get(c, "")
    return out


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    option_sets = _load_option_sets()
    with open(SAMPLE_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    audit = [_build_row(r, option_sets) for r in rows if r["in_audit_50"] == "True"]
    rest = [_build_row(r, option_sets) for r in rows if r["in_audit_50"] != "True"]

    _write(OUT_50, audit)
    _write(OUT_450, rest)

    def dist(label: str, rs: list[dict]) -> None:
        q = Counter(x["question"] for x in rs)
        n = len(rs) or 1
        share = ", ".join(f"{k}={v} ({v / n * 100:.1f}%)" for k, v in sorted(q.items()))
        print(f"  {label:>4}: {len(rs):>3} rows | {share}")

    print(f"Wrote {OUT_50.relative_to(REPO_ROOT)}  ({len(audit)} rows, {len(FIELDNAMES)} cols)")
    print(f"Wrote {OUT_450.relative_to(REPO_ROOT)}  ({len(rest)} rows, {len(FIELDNAMES)} cols)")
    print("Question representation (should track across all three):")
    dist("500", rows)
    dist("50", audit)
    dist("450", rest)


if __name__ == "__main__":
    main()
