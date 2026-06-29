"""Progress check for a running direct-probing experiment.

Usage:
    python experiments/show_progress_direct_probing.py           # auto-detect latest run
    python experiments/show_progress_direct_probing.py <RUN_TAG> # e.g. direct_complete001
    python experiments/show_progress_direct_probing.py --watch   # refresh every 10 s
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

SAMPLE_FRACTION = 0.20  # must match run_direct_probing.py


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "config" / "inference.example.yaml").exists():
            return p
    return Path.cwd()


def _compute_total(root: Path) -> int:
    personas_path = root / "src" / "generate_backgrounds" / "data" / "personas" / "personas.jsonl"
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    with open(personas_path) as f:
        for line in f:
            p = json.loads(line)
            gender = p["persona"].get("Gender")
            race = p["persona"].get("Race")
            if gender and race:
                grouped[(gender, race)] += 1
    return sum(max(1, round(n * SAMPLE_FRACTION)) for n in grouped.values())


def _count_rows(csv_path: Path) -> tuple[int, int]:
    """Return (success_rows, total_rows) for a stage-1 CSV."""
    try:
        success = 0
        total = 0
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            header = f.readline()
            # find a model response column (not prompt_id / prompt / prompt_metadata)
            for line in f:
                total += 1
                if '""status"":""success""' in line or '"status":"success"' in line:
                    success += 1
        return success, total
    except OSError:
        return 0, 0


def _count_stage2_success(csv_path: Path) -> tuple[int, int]:
    """Return (success_rows, total_rows) for a judgments CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        total = len(df)
        success = int((df["status"] == "success").sum()) if "status" in df.columns else 0
        return success, total
    except Exception:
        return 0, 0


def _find_stage1_csv(root: Path, run_tag: str | None) -> Path | None:
    """Find the most recent stage-1 CSV, optionally filtered by run_tag."""
    candidates = []
    for d in root.glob("logs/**/"):
        if any(part for part in d.parts if _matches(d.name, run_tag)):
            candidates += list(d.glob("*.csv"))
    # fallback: search all logs
    if not candidates:
        candidates = [
            p for p in root.glob("logs/**/*.csv")
            if "stage1" in str(p) and "direct-probing" in str(p)
            and (run_tag is None or run_tag in str(p))
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _matches(name: str, run_tag: str | None) -> bool:
    return "direct-probing" in name and "stage1" in name and (run_tag is None or run_tag in name)


def _find_stage2_csv(root: Path, run_tag: str | None) -> Path | None:
    candidates = [
        p for p in root.glob("logs/**/*.csv")
        if "stage2" in str(p) and "direct-probing" in str(p)
        and (run_tag is None or run_tag in str(p))
        and "orphan" not in p.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _age_str(age_s: float) -> str:
    if age_s < 120:
        return f"{age_s:.0f}s ago"
    return f"{age_s/60:.1f}m ago"


def _status_str(age_s: float) -> str:
    if age_s < 5:
        return "ACTIVE"
    if age_s < 60:
        return "likely active"
    return "idle / finished?"


def _report(root: Path, run_tag: str | None, total: int) -> None:
    s1 = _find_stage1_csv(root, run_tag)
    s2 = _find_stage2_csv(root, run_tag)

    print(f"Run tag : {run_tag or '(latest)'}")
    print(f"Total   : {total} subjects")
    print()

    if s1:
        ok, _ = _count_rows(s1)
        age_s = time.time() - s1.stat().st_mtime
        pct = ok / total * 100 if total else 0
        try:
            display = s1.relative_to(root)
        except ValueError:
            display = s1
        print(f"Stage 1 : {ok} / {total}  ({pct:.1f}%)  [{_status_str(age_s)}, last write {_age_str(age_s)}]")
        print(f"          {display}")
    else:
        print("Stage 1 : not found")

    print()

    if s2:
        ok, _ = _count_stage2_success(s2)
        age_s = time.time() - s2.stat().st_mtime
        pct = ok / total * 100 if total else 0
        try:
            display = s2.relative_to(root)
        except ValueError:
            display = s2
        print(f"Stage 2 : {ok} / {total}  ({pct:.1f}%)  [{_status_str(age_s)}, last write {_age_str(age_s)}]")
        print(f"          {display}")
    else:
        print("Stage 2 : not started yet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show direct-probing experiment progress.")
    parser.add_argument("run_tag", nargs="?", help="RUN_TAG to filter (e.g. direct_complete001)")
    parser.add_argument("--watch", action="store_true", help="Refresh every 10 seconds")
    args = parser.parse_args()

    root = _repo_root()
    total = _compute_total(root)

    interval = 10
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        _report(root, args.run_tag, total)
        if not args.watch:
            break
        print(f"\n(refreshing every {interval}s — Ctrl-C to stop)")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
