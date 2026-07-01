"""Progress check for a running behavioral-audit experiment.

Usage:
    python experiments/behavioral_audit/status.py                  # auto-detect latest run
    python experiments/behavioral_audit/status.py <RUN_ID>         # e.g. full002
    python experiments/behavioral_audit/status.py <RUN_ID> --watch # refresh every 10 s
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "config" / "inference.example.yaml").exists():
            return p
    return Path.cwd()


def _find_run_ids(root: Path) -> list[str]:
    """Return all run IDs found in logs/, sorted by most-recently-modified file."""
    logs = root / "logs"
    run_ids: dict[str, float] = {}
    # stage1 dirs: logs/behavioral-audit-<run_id>-q{1,2}-stage1/
    for d in logs.glob("behavioral-audit-*-stage1"):
        parts = d.name.split("-")
        # name format: behavioral-audit-<run_id>-q1-stage1
        # run_id is everything between "audit-" and "-q{n}-stage1"
        try:
            q_idx = next(i for i, p in enumerate(parts) if p.startswith("q") and p[1:].isdigit())
            run_id = "-".join(parts[2:q_idx])
            mtime = max((f.stat().st_mtime for f in d.glob("*.csv")), default=0.0)
            run_ids[run_id] = max(run_ids.get(run_id, 0.0), mtime)
        except StopIteration:
            continue
    return sorted(run_ids, key=lambda r: run_ids[r], reverse=True)


def _detect_models(root: Path, run_id: str) -> list[str]:
    """Infer the model list from the stage1 CSV header."""
    for q in ("q1", "q2"):
        d = root / "logs" / f"behavioral-audit-{run_id}-{q}-stage1"
        csvs = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime) if d.exists() else []
        if not csvs:
            continue
        with open(csvs[-1]) as f:
            header = f.readline().strip().split(",")
        skip = {"prompt_id", "prompt", "prompt_metadata", ""}
        models = [h.strip('"') for h in header if h.strip('"') not in skip]
        if models:
            return models
    return []


def _age_str(age_s: float) -> str:
    if age_s < 120:
        return f"{age_s:.0f}s ago"
    return f"{age_s / 60:.1f}m ago"


def _status_str(age_s: float) -> str:
    if age_s < 5:
        return "ACTIVE"
    if age_s < 60:
        return "likely active"
    return "idle / finished?"


def _report_stage1(root: Path, run_id: str, models: list[str]) -> None:
    for q in ("q1", "q2"):
        d = root / "logs" / f"behavioral-audit-{run_id}-{q}-stage1"
        if not d.exists():
            print(f"  stage1 {q}: not started")
            continue
        csvs = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            print(f"  stage1 {q}: dir exists but no CSVs yet")
            continue
        p = csvs[-1]
        age = time.time() - os.path.getmtime(p)
        with open(p) as f:
            rows = list(csv.DictReader(f))
        statuses: Counter = Counter()
        per_model: dict[str, Counter] = {m: Counter() for m in models}
        for r in rows:
            for model in models:
                v = r.get(model, "")
                if not v:
                    statuses["pending"] += 1
                    per_model[model]["pending"] += 1
                    continue
                try:
                    s = json.loads(v).get("status", "unknown")
                    statuses[s] += 1
                    per_model[model][s] += 1
                except Exception:
                    pass
        total = sum(statuses.values())
        success = statuses.get("success", 0)
        pct = 100 * success / total if total else 0
        model_parts = "  ".join(
            f"{m.split('_')[0]}: {per_model[m].get('success', 0)}/{sum(per_model[m].values())}"
            f" ({100 * per_model[m].get('success', 0) / sum(per_model[m].values()):.0f}%)"
            for m in models
            if sum(per_model[m].values()) > 0
        )
        status_label = _status_str(age)
        print(
            f"  stage1 {q}: {success}/{total} ({pct:.0f}%)"
            f"  [{model_parts}]"
            f"  {status_label}, last write {_age_str(age)}"
        )


def _report_stage2(root: Path, run_id: str) -> None:
    judges_dir = root / "logs" / "judges" / "behavioral-audit"
    for q in ("q1", "q2"):
        p = judges_dir / f"behavioral-audit-{run_id}-{q}-stage2.judgments.csv"
        if not p.exists():
            print(f"  stage2 {q}: not started")
            continue
        age = time.time() - os.path.getmtime(p)
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  stage2 {q}: file empty")
            continue
        current_hash = Counter(r["judge_config_hash"] for r in rows).most_common(1)[0][0]
        by_subject: dict = {}
        for r in rows:
            if r["judge_config_hash"] != current_hash:
                continue
            key = (r["subject_id"], r["subject_model_alias"])
            if key not in by_subject or r["completed_at"] > by_subject[key]["completed_at"]:
                by_subject[key] = r
        current = list(by_subject.values())
        success = sum(1 for r in current if r["status"] == "success")
        total = len(current)
        pct = 100 * success / total if total else 0
        fail_status = Counter(r["status"] for r in current if r["status"] != "success")
        fail_reasons = Counter(
            r.get("error_message", "")[:60] for r in current if r["status"] != "success"
        )
        fail_str = "  ".join(f"{s}:{n}" for s, n in fail_status.items())
        reason_str = "  ".join(f'{n}x "{reason}"' for reason, n in fail_reasons.most_common(3))
        status_label = _status_str(age)
        print(
            f"  stage2 {q}: {success}/{total} ({pct:.0f}%)"
            + (f"  fails: {fail_str}" if fail_str else "")
            + f"  {status_label}, last write {_age_str(age)}"
        )
        if reason_str:
            print(f"           reasons: {reason_str}")


def _report(root: Path, run_id: str) -> None:
    models = _detect_models(root, run_id)
    print(f"Run     : {run_id}")
    print(f"Models  : {', '.join(m.split('_')[0] for m in models) or '(unknown)'}")
    print()
    _report_stage1(root, run_id, models)
    print()
    _report_stage2(root, run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show behavioral-audit experiment progress.")
    parser.add_argument("run_id", nargs="?", help="Run ID to inspect (e.g. full002). Defaults to latest.")
    parser.add_argument("--watch", action="store_true", help="Refresh every 10 seconds.")
    args = parser.parse_args()

    root = _repo_root()

    if args.run_id:
        run_id = args.run_id
    else:
        available = _find_run_ids(root)
        if not available:
            print("No behavioral-audit runs found under logs/.")
            return
        run_id = available[0]

    interval = 10
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        _report(root, run_id)
        if not args.watch:
            break
        print(f"\n(refreshing every {interval}s — Ctrl-C to stop)")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
