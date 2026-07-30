"""Progress check for a raw stage1 CSV file (e.g. piped output or a saved file).

Usage:
    python experiments/behavioral_audit/status_csv.py <CSV_FILE>
    python experiments/behavioral_audit/status_csv.py <CSV_FILE> --watch
"""

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path


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


def _report(csv_path: Path) -> None:
    age = time.time() - os.path.getmtime(csv_path)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("File is empty.")
        return

    skip = {"prompt_id", "prompt", "prompt_metadata", ""}
    models = [h for h in rows[0] if h not in skip]

    print(f"File    : {csv_path}")
    print(f"Models  : {', '.join(m.split('_')[0] for m in models) or '(unknown)'}")
    print(f"Rows    : {len(rows)}")
    print()

    statuses = Counter()  # type: Counter
    per_model = {m: Counter() for m in models}  # type: dict

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
                statuses["unknown"] += 1
                per_model[model]["unknown"] += 1

    total = sum(statuses.values())
    success = statuses.get("success", 0)
    pct = 100 * success / total if total else 0

    model_parts = "  ".join(
        f"{m.split('_')[0]}: {per_model[m].get('success', 0)}/{sum(per_model[m].values())}"
        f" ({100 * per_model[m].get('success', 0) / sum(per_model[m].values()):.0f}%)"
        for m in models
        if sum(per_model[m].values()) > 0
    )

    fail_status = Counter(s for s, n in statuses.items() if s != "success" for _ in range(n))
    fail_str = "  ".join(f"{s}:{n}" for s, n in fail_status.items())

    status_label = _status_str(age)
    print(
        f"  total: {success}/{total} ({pct:.0f}%)"
        + (f"  fails: {fail_str}" if fail_str else "")
        + f"  {status_label}, last write {_age_str(age)}"
    )
    print(f"  [{model_parts}]")

    # top failure reasons
    fail_reasons: Counter = Counter()
    for r in rows:
        for model in models:
            v = r.get(model, "")
            if not v:
                continue
            try:
                parsed = json.loads(v)
                if parsed.get("status") != "success":
                    reason = parsed.get("error_message", "")[:60]
                    fail_reasons[reason] += 1
            except Exception:
                pass
    if fail_reasons:
        reason_str = "  ".join(f'{n}x "{r}"' for r, n in fail_reasons.most_common(3))
        print(f"  reasons: {reason_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show progress for a raw stage1 CSV file.")
    parser.add_argument("csv_file", help="Path to the stage1 CSV file.")
    parser.add_argument("--watch", action="store_true", help="Refresh every 10 seconds.")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    interval = 10
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        _report(csv_path)
        if not args.watch:
            break
        print(f"\n(refreshing every {interval}s — Ctrl-C to stop)")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
