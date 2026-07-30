"""Progress check for a running direct-probing experiment (multi-model aware).

Usage:
    python experiments/show_progress_direct_probing.py           # auto-detect latest run
    python experiments/show_progress_direct_probing.py <RUN_TAG> # e.g. direct_multimodel001
    python experiments/show_progress_direct_probing.py --watch   # refresh every 10 s

Handles both single-model and multi-model runs:
  * Stage 1 is one prompt x model matrix CSV. Progress is reported per model
    column (success cells / total rows), since each model is a separate column.
  * Stage 2 writes one judgments CSV per model
    (direct-probing-combined-<tag>-<model>-stage2). All of them are discovered
    and reported, not just the most recent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Columns in a stage-1 matrix CSV that are NOT model response columns.
_NON_MODEL_COLS = {"prompt_id", "prompt", "prompt_metadata"}


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "config" / "inference.example.yaml").exists():
            return p
    return Path.cwd()


def _stage1_progress(csv_path: Path) -> tuple[int, dict[str, int]]:
    """Parse a stage-1 matrix CSV.

    Returns (total_rows, {model_alias: success_count}). A cell counts as a
    success when its JSON payload has status == "success".
    """
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception:
        return 0, {}

    total = len(df)
    model_cols = [c for c in df.columns if c not in _NON_MODEL_COLS]
    per_model: dict[str, int] = {}
    for col in model_cols:
        ok = 0
        for v in df[col]:
            if not isinstance(v, str) or not v:
                continue
            try:
                if json.loads(v).get("status") == "success":
                    ok += 1
            except (json.JSONDecodeError, TypeError):
                continue
        per_model[col] = ok
    return total, per_model


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
    candidates = [
        p
        for p in root.glob("logs/**/*.csv")
        if "stage1" in str(p) and "direct-probing" in str(p)
        and (run_tag is None or run_tag in str(p))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_stage2_csvs(root: Path, run_tag: str | None) -> list[Path]:
    """Find ALL stage-2 judgments CSVs for the run (one per model).

    Multi-model runs write one file per model, distinguished by filename:
      direct-probing-combined-<tag>-<model>-stage2.judgments.csv
    Single-model runs have exactly one. Each distinct stage-2 experiment is
    identified by its filename stem (before the timestamp/suffix), and the most
    recent CSV per stem is kept.
    """
    candidates = [
        p
        for p in root.glob("logs/**/*.csv")
        if "stage2" in str(p) and "direct-probing" in str(p)
        and (run_tag is None or run_tag in str(p))
        and "orphan" not in p.name
    ]
    # keep the newest CSV per distinct stage-2 experiment (keyed by dir + the
    # filename up to and including "-stage2", so multiple models in the same
    # directory stay separate)
    by_key: dict[tuple[Path, str], Path] = {}
    for p in candidates:
        stem = p.name
        idx = stem.find("-stage2")
        key = (p.parent, stem[: idx + len("-stage2")] if idx != -1 else stem)
        cur = by_key.get(key)
        if cur is None or p.stat().st_mtime > cur.stat().st_mtime:
            by_key[key] = p
    return sorted(by_key.values(), key=lambda p: p.name)


def _derive_tag(s1: Path) -> str | None:
    """Recover the run tag from a stage-1 CSV path so stage-2 lookup is scoped
    to the same run (path looks like logs/direct-probing-<tag>-stage1/<ts>.csv)."""
    for part in s1.parts:
        if "direct-probing" in part and "stage1" in part:
            return part.removeprefix("direct-probing-combined-").removeprefix(
                "direct-probing-"
            ).removesuffix("-stage1")
    return None


def _stage2_model_label(csv_path: Path, tag: str | None) -> str:
    """Best-effort model name for a stage-2 CSV.

    Files are named direct-probing-combined-<tag>-<model>-stage2.judgments.csv
    (multi-model) or ...-<tag>-stage2... (single-model). Fall back to the parent
    directory name for the older layout that used one dir per experiment.
    """
    name = csv_path.name
    idx = name.find("-stage2")
    stem = name[:idx] if idx != -1 else csv_path.parent.name
    inner = stem.removeprefix("direct-probing-combined-").removeprefix("direct-probing-")
    if tag and inner.startswith(tag + "-"):
        return inner[len(tag) + 1 :]
    if tag and inner == tag:
        return "(single-model)"
    return inner or "(single-model)"


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


def _pct(ok: int, total: int) -> float:
    return ok / total * 100 if total else 0.0


def _report(root: Path, run_tag: str | None) -> None:
    s1 = _find_stage1_csv(root, run_tag)

    # If no run_tag was given, derive it from the stage-1 CSV path so stage-2
    # lookup is scoped to the same run (avoids showing a stale run's stage-2).
    effective_tag = run_tag if run_tag is not None else (_derive_tag(s1) if s1 else None)

    print(f"Run tag : {run_tag or effective_tag or '(latest)'}")

    if s1:
        total, per_model = _stage1_progress(s1)
        age_s = time.time() - s1.stat().st_mtime
        try:
            display = s1.relative_to(root)
        except ValueError:
            display = s1
        print(f"Total   : {total} subjects  |  {len(per_model)} model(s)")
        print()
        print(f"Stage 1 : [{_status_str(age_s)}, last write {_age_str(age_s)}]")
        print(f"          {display}")
        for model, ok in per_model.items():
            print(f"            {model:28s} {ok:5d} / {total}  ({_pct(ok, total):5.1f}%)")
    else:
        print("Total   : (unknown — stage-1 CSV not found)")
        print()
        print("Stage 1 : not found")
        total = 0

    print()

    s2_csvs = _find_stage2_csvs(root, effective_tag)
    if s2_csvs:
        print(f"Stage 2 : {len(s2_csvs)} judgments CSV(s)")
        for s2 in s2_csvs:
            ok, s2_total = _count_stage2_success(s2)
            age_s = time.time() - s2.stat().st_mtime
            label = _stage2_model_label(s2, effective_tag)
            denom = total or s2_total
            try:
                display = s2.relative_to(root)
            except ValueError:
                display = s2
            print(
                f"            {label:28s} {ok:5d} / {denom}  ({_pct(ok, denom):5.1f}%)"
                f"  [{_status_str(age_s)}, {_age_str(age_s)}]"
            )
            print(f"            {display}")
    else:
        print("Stage 2 : not started yet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show direct-probing experiment progress.")
    parser.add_argument("run_tag", nargs="?", help="RUN_TAG to filter (e.g. direct_multimodel001)")
    parser.add_argument("--watch", action="store_true", help="Refresh every 10 seconds")
    args = parser.parse_args()

    root = _repo_root()

    interval = 10
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        _report(root, args.run_tag)
        if not args.watch:
            break
        print(f"\n(refreshing every {interval}s — Ctrl-C to stop)")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
