"""Export a behavioral-audit run from the gitignored working area into a committable results dir.

While a run is in progress its Stage-1 artifacts live under ``logs/`` — gitignored,
so they never show up in ``git status``. This script copies the finished Stage-1
CSVs of one run-tag into

    experiments/behavioral_audit/results_behavioral_audit/results_<run-tag>/

which IS tracked, so the CSVs can be committed alongside the other models'
results (``results_full001/``, ``results_baseline/``, …).

What it copies (each part is optional — missing pieces are skipped, not fatal):
  - Stage-1 matrix CSVs + ``.meta.json`` sidecars  (logs/behavioral-audit-<tag>-q{1,2}-stage1/)
  - Stage-2 judgments CSVs (if present)             (logs/judges/behavioral-audit/*<tag>*stage2*.judgments.csv)
  - Eval figures (if present)                       (logs/judges/behavioral-audit/figures/*<tag>*eval*/)

Transient ``.lock`` files are skipped. A ``EXPORT_MANIFEST.json`` records the file
list and per-question success counts; the copy is verified (row / success counts
match the source).

Usage
    python experiments/behavioral_audit/export_results.py --run-tag full001-e2b \
        --subject-alias gemma-4-e2b_modal
    python experiments/behavioral_audit/export_results.py --run-tag full001-ministral3-8b \
        --subject-alias ministral-3-8b_modal
    # then:  git add experiments/behavioral_audit/results_behavioral_audit/results_<tag> && git commit
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "config" / "inference.example.yaml").exists():
            return p
    return Path.cwd()


REPO_ROOT = _repo_root()
LOGS_DIR = REPO_ROOT / "logs"
JUDGE_LOGS_DIR = LOGS_DIR / "judges" / "behavioral-audit"
RESULTS_PARENT = REPO_ROOT / "experiments" / "behavioral_audit" / "results_behavioral_audit"


def _success_count(csv_path: Path, alias: str | None) -> tuple[int, int]:
    """(success_cells, rows) for a Stage-1 matrix CSV, counting the alias column."""
    df = pd.read_csv(csv_path)
    if alias and alias in df.columns:
        s = int(df[alias].apply(lambda x: isinstance(x, str) and '"success"' in x).sum())
        return s, len(df)
    return 0, len(df)


def _copy_stage1(experiment_name: str, dest: Path, alias: str | None) -> list[dict]:
    exported: list[dict] = []
    for q in ("q1", "q2"):
        src = LOGS_DIR / f"{experiment_name}-{q}-stage1"
        if not src.exists():
            print(f"  [{q}] stage1: none at {src.relative_to(REPO_ROOT)} — skip")
            continue
        ddir = dest / f"{experiment_name}-{q}-stage1"
        ddir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for f in sorted(src.iterdir()):
            if f.suffix == ".lock" or not f.is_file():
                continue  # skip transient lock files; copy CSV + .meta.json
            if f.name.endswith(".csv") or f.name.endswith(".meta.json"):
                shutil.copy2(f, ddir / f.name)
                files.append(f.name)
        csvs = sorted(ddir.glob("*.csv"))
        succ, rows = _success_count(csvs[-1], alias) if csvs else (0, 0)
        # verify the copy matches the source csv byte-for-byte count
        if csvs:
            src_succ, src_rows = _success_count(sorted(src.glob("*.csv"))[-1], alias)
            assert (succ, rows) == (src_succ, src_rows), f"copy mismatch for {q}"
        print(
            f"  [{q}] stage1: {len(files)} files -> "
            f"{ddir.relative_to(REPO_ROOT)}  ({succ}/{rows} {alias or 'cells'} success)"
        )
        exported.append(
            {"question": q, "kind": "stage1", "files": files, "success": succ, "rows": rows}
        )
    return exported


def _copy_matching(patterns: list[str], search_dir: Path, dest: Path, label: str) -> list[dict]:
    """Copy files/dirs in search_dir whose name matches any glob pattern."""
    out: list[dict] = []
    if not search_dir.exists():
        return out
    seen: set[Path] = set()
    for pat in patterns:
        for match in sorted(search_dir.glob(pat)):
            if match in seen:
                continue
            seen.add(match)
            target = dest / match.name
            if match.is_dir():
                shutil.copytree(match, target, dirs_exist_ok=True)
                n = sum(1 for _ in target.rglob("*") if _.is_file())
                print(
                    f"  {label}: dir {match.name}/ ({n} files) -> {target.relative_to(REPO_ROOT)}"
                )
                out.append({"kind": label, "name": match.name, "files": n})
            elif match.is_file():
                shutil.copy2(match, target)
                print(f"  {label}: {match.name} -> {target.relative_to(REPO_ROOT)}")
                out.append({"kind": label, "name": match.name})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-tag", required=True, help="run tag, e.g. full001-e2b")
    ap.add_argument(
        "--subject-alias",
        default=None,
        help="model column to report success counts for (e.g. gemma-4-e2b_modal)",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="override destination (default experiments/behavioral_audit/results_behavioral_audit/results_<tag>)",
    )
    args = ap.parse_args()

    experiment_name = f"behavioral-audit-{args.run_tag}"
    dest = args.dest or (RESULTS_PARENT / f"results_{args.run_tag}")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Exporting run '{args.run_tag}' -> {dest.relative_to(REPO_ROOT)}\n")

    manifest: dict = {
        "run_tag": args.run_tag,
        "subject_alias": args.subject_alias,
        "experiment_name": experiment_name,
        "exported": [],
    }

    manifest["exported"] += _copy_stage1(experiment_name, dest, args.subject_alias)
    # Stage-2 judgments: canonical (…-stage2.judgments.csv) or modal (…-stage2-modal.judgments.csv)
    manifest["exported"] += _copy_matching(
        [f"{experiment_name}-q*-stage2*.judgments.csv"], JUDGE_LOGS_DIR, dest, "stage2"
    )
    # Eval figures: any figures dir named like <tag>-eval / <tag>-modal-eval
    manifest["exported"] += _copy_matching(
        [f"*{args.run_tag}*eval*"], JUDGE_LOGS_DIR / "figures", dest, "figures"
    )

    (dest / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    n_stage1 = sum(1 for e in manifest["exported"] if e["kind"] == "stage1")
    n_stage2 = sum(1 for e in manifest["exported"] if e["kind"] == "stage2")
    print(f"\nDone. stage1={n_stage1} question(s), stage2={n_stage2} judgment file(s).")
    print(f"Committable results at: {dest.relative_to(REPO_ROOT)}")
    print(f"  git add {dest.relative_to(REPO_ROOT)} && git commit -m 'Add {args.run_tag} results'")
    if n_stage2 == 0:
        print(
            "  (Stage-2 judgments not found yet — run run_behavioral_audit.py "
            "STAGE2_ONLY=True, then re-run this export.)"
        )


if __name__ == "__main__":
    main()
