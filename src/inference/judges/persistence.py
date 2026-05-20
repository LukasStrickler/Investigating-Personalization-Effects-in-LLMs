"""Atomic CSV persistence + resume lookup for judge runs.

Write strategy: keep the full in-memory row table; after each completed verdict, write
the entire table to a tempfile and ``os.replace()`` it over the destination. This is
durable on POSIX and matches the "abort and resume without data loss" requirement.

For multi-worker safety inside one process, all writes go through an ``asyncio.Lock``
that the runner injects into the writer. We do not provide inter-process locking; if
two processes target the same CSV, last write wins.
"""

from __future__ import annotations

import contextlib
import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from inference.judges.csv_schema import (
    COLUMNS,
    csv_writer_kwargs,
    resume_key,
    row_to_verdict,
    verdict_to_row,
)
from inference.judges.types import JudgeStatus, JudgeVerdict


class JudgeCSVCorruptError(RuntimeError):
    pass


class JudgmentCSVWriter:
    """Single-file judgments CSV writer with full-table atomic rewrites."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = Path(csv_path)
        # In-memory ordered row store keyed by judgment_id.
        self._rows: dict[str, dict[str, str]] = {}
        # Resume index: (subject_id, subject_model_alias, judge_alias, judge_config_hash) -> status
        self._resume_index: dict[tuple[str, str, str, str], JudgeStatus] = {}

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def load(self) -> None:
        """Load existing CSV state. Safe to call on a non-existent file."""
        if not self._csv_path.exists():
            return
        with self._csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, **csv_writer_kwargs())
            if reader.fieldnames is None:
                return
            missing = [c for c in COLUMNS if c not in reader.fieldnames]
            if missing:
                raise JudgeCSVCorruptError(
                    f"Judgments CSV {self._csv_path} missing columns: {missing}"
                )
            for row in reader:
                jid = row.get("judgment_id") or ""
                if not jid:
                    continue
                self._rows[jid] = {k: row.get(k, "") for k in COLUMNS}
                try:
                    status = JudgeStatus(row.get("status", JudgeStatus.NOT_REQUESTED.value))
                except ValueError as exc:
                    raise JudgeCSVCorruptError(
                        f"Unknown status {row.get('status')!r} in {self._csv_path}"
                    ) from exc
                key = resume_key(
                    row.get("subject_id", ""),
                    row.get("subject_model_alias") or None,
                    row.get("judge_alias", ""),
                    row.get("judge_config_hash", ""),
                )
                self._resume_index[key] = status

    def initialize(self) -> None:
        """Create the file (with header) if it doesn't exist; otherwise no-op."""
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            self._atomic_rewrite()

    def is_completed(
        self,
        *,
        subject_id: str,
        subject_model_alias: str | None,
        judge_alias: str,
        judge_config_hash: str,
    ) -> bool:
        """A cell is 'completed' for resume purposes iff its row exists with SUCCESS."""
        key = resume_key(subject_id, subject_model_alias, judge_alias, judge_config_hash)
        return self._resume_index.get(key) is JudgeStatus.SUCCESS

    def upsert(self, verdict: JudgeVerdict) -> None:
        """Add or replace a row and atomically persist the entire table."""
        row = verdict_to_row(verdict)
        # Drop any prior row with the same resume key (e.g. previous failed attempt).
        key = resume_key(
            verdict.subject_id,
            verdict.subject_model_alias,
            verdict.judge_alias,
            verdict.judge_config_hash,
        )
        for existing_jid, existing_row in list(self._rows.items()):
            existing_key = resume_key(
                existing_row.get("subject_id", ""),
                existing_row.get("subject_model_alias") or None,
                existing_row.get("judge_alias", ""),
                existing_row.get("judge_config_hash", ""),
            )
            if existing_key == key and existing_jid != verdict.judgment_id:
                self._rows.pop(existing_jid, None)
        self._rows[verdict.judgment_id] = row
        self._resume_index[key] = verdict.status
        self._atomic_rewrite()

    def all_verdicts(self) -> list[JudgeVerdict]:
        return [row_to_verdict(r) for r in self._rows.values()]

    def _atomic_rewrite(self) -> None:
        target_dir = self._csv_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp = NamedTemporaryFile(  # noqa: SIM115 - manually managed for atomic replace
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(target_dir),
            prefix=f".{self._csv_path.name}.tmp.",
            delete=False,
        )
        try:
            writer = csv.DictWriter(tmp, fieldnames=COLUMNS, **csv_writer_kwargs())
            writer.writeheader()
            for row in self._rows.values():
                writer.writerow(row)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
            tmp.close()
            os.replace(tmp_path, self._csv_path)
        except Exception:
            tmp.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp.name)
            raise


def default_csv_path(experiment_name: str, output_dir: Path | None) -> Path:
    base = output_dir if output_dir is not None else Path("logs/judges")
    return base / f"{experiment_name}.judgments.csv"


__all__ = [
    "JudgeCSVCorruptError",
    "JudgmentCSVWriter",
    "default_csv_path",
]
