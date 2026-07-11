"""Audit the generated human-review CSVs and report their distribution.

Runs a set of hard integrity checks (schema, counts, disjointness, no empty
required fields, option-set consistency, human columns still blank) and a set of
soft warnings (judge labels outside the option set, none-declared rate), then
prints a distribution table for the 50 / 450 / 500 splits across every axis we
stratified on so we can eyeball representativeness.

Exit code is non-zero iff a hard check fails, so it doubles as a CI gate.

Usage:
    .venv/bin/python experiments/behavioral_audit/judge_audit/audit_human_review.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from pathlib import Path

from build_human_review import (  # keep the schema in one place
    CONTEXT_COLS,
    FIELDNAMES,
    HUMAN_COLS,
    NONE_SENTINEL,
    PROVENANCE_COLS,
    REVIEW_CORE_COLS,
)
from prepare_judge_audit_sample import (
    CANONICAL_PROBES,
    INPUT_STRATUM_AXES,
    JUDGE_ALIAS,
    RESEARCH_RUNS,
    RUN_MODELS,
    filter_research_population,
    load_population,
)

csv.field_size_limit(10**7)

HERE = Path(__file__).resolve().parent
CSV_50 = HERE / "judge_audit_human_50.csv"
CSV_450 = HERE / "judge_audit_human_450.csv"
POOL = HERE / "judge_audit_sample_500.csv"

# Fields a reviewer must be able to read on every row.
REQUIRED_NONEMPTY = [
    "judgment_id", "question", "probe_question", "subject_response",
    "options", "n_options", "final_class", "raw_output",
]
# Input axes the sample was stratified on (not judge outcomes).
DIST_AXES = [
    "question", "run_tag", "subject_model_alias", "true_gender", "true_race",
]

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def ok(msg: str) -> None:
    print(f"  {_c('PASS', '32')}  {msg}")


def warn(msg: str, warns: list[str]) -> None:
    warns.append(msg)
    print(f"  {_c('WARN', '33')}  {msg}")


def fail(msg: str, fails: list[str]) -> None:
    fails.append(msg)
    print(f"  {_c('FAIL', '31')}  {msg}")


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _opts(row: dict) -> list[str]:
    return [o.strip() for o in row["options"].split(" | ")]


def _pct(rows: list[dict], col: str) -> dict[str, float]:
    n = len(rows) or 1
    return {k: v / n * 100 for k, v in Counter(r.get(col, "") for r in rows).items()}


def _max_drift(a: list[dict], b: list[dict], col: str) -> float:
    pa, pb = _pct(a, col), _pct(b, col)
    keys = set(pa) | set(pb)
    return max((abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys), default=0.0)


def check_integrity(a: list[dict], b: list[dict], header_a: list[str], header_b: list[str]) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    warns: list[str] = []
    print(_c("Integrity checks", "1"))

    # 1. schema
    for name, hdr in (("50", header_a), ("450", header_b)):
        if hdr == FIELDNAMES:
            ok(f"{name}: header matches expected {len(FIELDNAMES)}-column schema")
        else:
            fail(f"{name}: header mismatch (got {len(hdr)} cols) — {set(hdr) ^ set(FIELDNAMES)}", fails)

    # 2. counts / identity
    if len(a) == 50:
        ok("50-file has exactly 50 rows")
    else:
        fail(f"50-file has {len(a)} rows, expected 50", fails)
    if len(b) == 450:
        ok("450-file has exactly 450 rows")
    else:
        fail(f"450-file has {len(b)} rows, expected 450", fails)

    ids_a = [r["judgment_id"] for r in a]
    ids_b = [r["judgment_id"] for r in b]
    allids = ids_a + ids_b
    if len(set(allids)) == len(allids):
        ok(f"all {len(allids)} judgment_ids unique")
    else:
        fail(f"{len(allids) - len(set(allids))} duplicate judgment_id(s)", fails)
    if set(ids_a).isdisjoint(ids_b):
        ok("50 and 450 are disjoint")
    else:
        fail(f"{len(set(ids_a) & set(ids_b))} id(s) appear in both files", fails)

    if POOL.exists():
        pool_ids = {r["judgment_id"] for r in _load(POOL)}
        if set(allids) == pool_ids:
            ok("50 ∪ 450 == the 500-row source pool exactly")
        else:
            fail(f"union != pool (missing {len(pool_ids - set(allids))}, extra {len(set(allids) - pool_ids)})", fails)

    rows = a + b

    judges = Counter(r.get("judge_alias", "") for r in rows)
    if judges == {JUDGE_ALIAS: len(rows)}:
        ok(f"every row uses production judge {JUDGE_ALIAS}")
    else:
        fail(f"unexpected judge_alias mix: {dict(judges)} — rebuild with prepare_judge_audit_sample.py", fails)

    bad_strata = [
        r["judgment_id"] for r in rows
        if len((r.get("stratum") or "").split("|")) != len(INPUT_STRATUM_AXES)
    ]
    if not bad_strata:
        ok(f"stratum uses {len(INPUT_STRATUM_AXES)} input axes (no outcome label in key)")
    else:
        fail(f"{len(bad_strata)} row(s) with non-input stratum format", fails)

    out_of_scope = [r["judgment_id"] for r in rows if r["run_tag"] not in RESEARCH_RUNS]
    if not out_of_scope:
        ok(f"every row is from a research run ({', '.join(sorted(RESEARCH_RUNS))})")
    else:
        fail(f"{len(out_of_scope)} row(s) outside research runs", fails)

    bad_probe = [
        (r["judgment_id"], r["question"])
        for r in rows
        if (r.get("probe_question") or "").strip() != CANONICAL_PROBES.get(r["question"], "")
    ]
    if not bad_probe:
        ok("probe_question matches canonical text for every question type")
    else:
        fail(f"{len(bad_probe)} row(s) with non-canonical probe_question: {bad_probe[:3]}", fails)

    bad_model = [
        (r["judgment_id"], r["run_tag"], r["subject_model_alias"])
        for r in rows
        if r["subject_model_alias"] not in RUN_MODELS.get(r["run_tag"], frozenset())
    ]
    if not bad_model:
        ok("subject model matches the run it belongs to on every row")
    else:
        fail(f"{len(bad_model)} row(s) with wrong model for run: {bad_model[:3]}", fails)

    bad_q = [
        (r["judgment_id"], r["run_tag"], r["question"])
        for r in rows
        if r["question"] not in RESEARCH_RUNS.get(r["run_tag"], frozenset())
    ]
    if not bad_q:
        ok("question type matches run on every row")
    else:
        fail(f"{len(bad_q)} row(s) with wrong question for run: {bad_q[:3]}", fails)

    # 3. required fields non-empty
    empties = {c: sum(1 for r in rows if not (r.get(c, "") or "").strip()) for c in REQUIRED_NONEMPTY}
    bad = {c: n for c, n in empties.items() if n}
    if not bad:
        ok(f"no empty values in required fields ({', '.join(REQUIRED_NONEMPTY)})")
    else:
        fail(f"empty required fields: {bad}", fails)

    # 4. n_options consistency + option set stable per question
    bad_n = sum(1 for r in rows if str(len(_opts(r))) != r["n_options"])
    if bad_n == 0:
        ok("n_options matches the option list on every row")
    else:
        fail(f"{bad_n} row(s) where n_options != len(options)", fails)
    for q in sorted({r["question"] for r in rows}):
        distinct = {r["options"] for r in rows if r["question"] == q}
        n = next(r["n_options"] for r in rows if r["question"] == q)
        if len(distinct) == 1:
            ok(f"question={q}: single stable option set ({n} options)")
        else:
            fail(f"question={q}: {len(distinct)} different option strings", fails)

    # 5. human-review columns still blank (ready to fill)
    dirty = {c: sum(1 for r in rows if (r.get(c, "") or "").strip()) for c in HUMAN_COLS}
    filled = {c: n for c, n in dirty.items() if n}
    if not filled:
        ok(f"all {len(HUMAN_COLS)} human-review columns blank and ready")
    else:
        fail(f"human-review columns already contain data: {filled}", fails)

    # --- soft checks ---
    print(_c("\nSoft checks", "1"))
    outside = [(r["judgment_id"], r["question"], r["final_class"]) for r in rows
               if r["final_class"].strip() and r["final_class"].strip() not in _opts(r)]
    if outside:
        warn(f"{len(outside)} judge label(s) outside the allowed option set "
             f"(faithful judge output — reviewers should mark these): {outside[:5]}", warns)
    else:
        ok("every judge label is a member of its option set")

    none_n = sum(1 for r in rows if r["none_declared"] == "True")
    none_sent = sum(1 for r in rows if r["final_class"].strip() == NONE_SENTINEL)
    if none_n == none_sent:
        ok(f"none_declared consistent: {none_n} rows flagged and shown as {NONE_SENTINEL}")
    else:
        warn(f"none_declared={none_n} but {none_sent} rows show {NONE_SENTINEL}", warns)

    short = sum(1 for r in rows if len(r["subject_response"]) < 20)
    if short:
        warn(f"{short} row(s) with a very short (<20 char) subject_response", warns)
    else:
        ok("no suspiciously short subject responses")

    return fails, warns


def report_population_drift(sample: list[dict]) -> None:
    """Compare the 500-row sample against the research-scope population."""
    research, _ = filter_research_population(load_population())
    print(_c("\nSample vs research population  (500 sample / research pop, %)", "1"))
    print(_c("  Note: q1/q2 are intentionally 50/50 in the sample (research design);", "90"))
    print(_c("  the pool skews q2-heavy because full001 q1 judging was incomplete.", "90"))
    for axis in DIST_AXES:
        print(f"\n  {_c(axis, '36')}")
        share = _pct(research, axis)
        keys = sorted({r.get(axis, "") for r in research}, key=lambda k: -share.get(k, 0))
        pr, ps = _pct(research, axis), _pct(sample, axis)
        max_d = 0.0
        for k in keys:
            label = k if k != "" else "(empty)"
            d = ps.get(k, 0) - pr.get(k, 0)
            max_d = max(max_d, abs(d))
            print(f"    {label:<34} {ps.get(k,0):5.1f}  {pr.get(k,0):5.1f}  {d:+5.1f}pp")
        flag = _c("  <- drift>3pp", "33") if max_d > 3 else ""
        print(f"    {'max |drift|':<34} {'-':>5}  {'-':>5}  {max_d:5.1f}{flag}")

    print(f"\n  {_c('run × question × model (sample counts)', '36')}")
    cross = Counter((r["run_tag"], r["question"], r["subject_model_alias"]) for r in sample)
    for key in sorted(cross):
        rt, q, m = key
        n = cross[key]
        print(f"    {rt}|{q}|{m}: {n}")


def report_distribution(a: list[dict], b: list[dict]) -> None:
    full = a + b
    print(_c("\nDistribution across axes  (50 / 450 / 500, % of split)", "1"))
    for axis in DIST_AXES:
        print(f"\n  {_c(axis, '36')}")
        share = _pct(full, axis)
        keys = sorted({r.get(axis, "") for r in full}, key=lambda k: -share.get(k, 0))
        pa, pb, pf = _pct(a, axis), _pct(b, axis), _pct(full, axis)
        for k in keys:
            label = k if k != "" else "(empty)"
            print(f"    {label:<34} {pa.get(k,0):5.1f}  {pb.get(k,0):5.1f}  {pf.get(k,0):5.1f}")
        d50 = _max_drift(a, full, axis)
        d450 = _max_drift(b, full, axis)
        flag = _c("  <- drift>5pp on a level", "33") if max(d50, d450) > 5 else ""
        print(f"    {'max |drift vs 500|':<34} {d50:5.1f}  {d450:5.1f}   {'-':>4}{flag}")

    # Outcome axis — informational only; not used for stratification.
    print(f"\n  {_c('final_class (outcome — not stratified)', '36')}")
    for name, rows in (("50", a), ("450", b), ("500", full)):
        cc = Counter(r["final_class"] for r in rows)
        print(f"    {name:>4}: {len(cc)} distinct labels over {len(rows)} rows "
              f"(top: {', '.join(f'{k}×{v}' for k, v in cc.most_common(3))})")
    none_n = sum(1 for r in full if r["none_declared"] == "True")
    print(f"    none_declared: {none_n}/{len(full)} ({none_n / len(full) * 100:.1f}%)")

    # direct_probe demographic balance (small stratum, worth eyeballing).
    dp = [r for r in full if r["question"] == "direct_probe"]
    if dp:
        print(f"\n  {_c('direct_probe demographics', '36')} ({len(dp)} rows)")
        print(f"    true_gender: {dict(Counter(r['true_gender'] for r in dp))}")
        print(f"    true_race  : {dict(Counter(r['true_race'] for r in dp))}")


def main() -> int:
    for p in (CSV_50, CSV_450):
        if not p.exists():
            print(f"{_c('FAIL', '31')}  missing {p.name} — run build_human_review.py first")
            return 2

    with open(CSV_50, encoding="utf-8") as f:
        header_a = next(csv.reader(f))
    with open(CSV_450, encoding="utf-8") as f:
        header_b = next(csv.reader(f))
    a, b = _load(CSV_50), _load(CSV_450)

    print(_c(f"Auditing {CSV_50.name} ({len(a)}) + {CSV_450.name} ({len(b)})\n", "1"))
    fails, warns = check_integrity(a, b, header_a, header_b)
    report_population_drift(a + b)
    report_distribution(a, b)

    print(_c("\nSummary", "1"))
    print(f"  columns: {len(FIELDNAMES)}  "
          f"(context {len(CONTEXT_COLS)} · review {len(REVIEW_CORE_COLS)} · "
          f"human {len(HUMAN_COLS)} · provenance {len(PROVENANCE_COLS)})")
    if fails:
        print(f"  {_c('RESULT: FAIL', '31;1')} — {len(fails)} hard issue(s), {len(warns)} warning(s)")
        return 1
    print(f"  {_c('RESULT: PASS', '32;1')} — 0 hard issues, {len(warns)} warning(s) "
          f"(expected & benign)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
