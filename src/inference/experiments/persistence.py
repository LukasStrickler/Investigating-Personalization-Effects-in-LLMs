from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO

from inference.experiments.csv_schema import (
    MISSING_CELL,
    PROMPT_COLUMN,
    PROMPT_ID_COLUMN,
    SCHEMA_VERSION,
    CellStatus,
    MatrixCell,
    build_matrix_headers,
    build_sidecar_metadata,
    canonical_prompt_spec,
    compute_prompt_id,
    csv_writer_kwargs,
    metadata_sidecar_path,
    serialize_prompt_content,
)


def build_experiment_csv_path(experiment_name: str, base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else Path("logs")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return root / experiment_name / f"{timestamp}.csv"


def load_existing_matrix(
    csv_path: Path,
) -> tuple[set[str], dict[tuple[str, str], CellStatus]]:
    """Load existing CSV state for resume mode.

    Reads UTF-8 CSV (prompt_id, prompt, then one column per model). Only SUCCESS cells are completed for resume.

    Returns:
        - prompt_ids_seen: set of prompt_id values in the CSV (normalized, stripped)
        - completed_cells: dict mapping (prompt_id, alias) -> CellStatus for SUCCESS cells
    """
    prompt_ids_seen: set[str] = set()
    completed_cells: dict[tuple[str, str], CellStatus] = {}

    if not csv_path.exists():
        return prompt_ids_seen, completed_cells

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, **csv_writer_kwargs())
        fieldnames = reader.fieldnames or []
        aliases = [n for n in fieldnames if n not in (PROMPT_ID_COLUMN, PROMPT_COLUMN)]
        for row in reader:
            raw_pid = row.get(PROMPT_ID_COLUMN, "")
            prompt_id = raw_pid.strip() if isinstance(raw_pid, str) else ""
            if prompt_id:
                prompt_ids_seen.add(prompt_id)

            if not prompt_id:
                continue

            for alias in aliases:
                cell = MatrixCell.from_csv_cell(row.get(alias, MISSING_CELL) or MISSING_CELL)
                if cell is not None and cell.status is CellStatus.SUCCESS:
                    completed_cells[(prompt_id, alias)] = CellStatus.SUCCESS
    return prompt_ids_seen, completed_cells


class MatrixCSVWriter:
    """Writes experiment CSV: prompt_id, prompt, then one column per model. Cells hold status/response/error/metadata only.

    Single-writer per path (file locking). UTF-8. Schema version in sidecar.
    """

    def __init__(self, csv_path: Path, model_aliases: list[str]) -> None:
        self._csv_path = csv_path
        self._headers = build_matrix_headers(model_aliases)
        self._model_aliases = [
            h for h in self._headers if h not in (PROMPT_ID_COLUMN, PROMPT_COLUMN)
        ]
        self._lock_path = Path(f"{csv_path}.lock")

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def initialize(self, prompts: Sequence[str | dict[str, Any]]) -> None:
        rows = []
        for p in prompts:
            c = canonical_prompt_spec(p)
            rows.append(
                self._empty_row(
                    prompt_id=compute_prompt_id(c),
                    prompt=serialize_prompt_content(c),
                )
            )
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked_file():
            self._rewrite_rows(rows)
            self._write_sidecar()

    def load_existing_state(self) -> tuple[list[str], dict[str, dict[str, CellStatus]]]:
        prompt_ids: list[str] = []
        prompt_id_set: set[str] = set()
        completed_cells: dict[str, dict[str, CellStatus]] = {}

        if not self._csv_path.exists():
            return prompt_ids, completed_cells

        with self._csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file, **csv_writer_kwargs())
            if reader.fieldnames != self._headers:
                raise ValueError(f"Unexpected matrix CSV headers in {self._csv_path}")

            for row in reader:
                prompt_id = row.get("prompt_id", "")
                if not prompt_id:
                    continue
                if prompt_id not in prompt_id_set:
                    prompt_id_set.add(prompt_id)
                    prompt_ids.append(prompt_id)

                for alias in self._model_aliases:
                    raw_cell = row.get(alias, MISSING_CELL)
                    cell = MatrixCell.from_csv_cell(raw_cell)
                    if cell is not None and cell.status is CellStatus.SUCCESS:
                        completed_cells.setdefault(prompt_id, {})[alias] = CellStatus.SUCCESS

        return prompt_ids, completed_cells

    def append_missing_prompts(self, prompts: Sequence[str | dict[str, Any]]) -> list[str]:
        with self._locked_file():
            if not self._csv_path.exists():
                raise FileNotFoundError(f"Matrix CSV does not exist: {self._csv_path}")

            rows = self._read_rows()
            existing_prompt_ids = {row["prompt_id"] for row in rows if row.get("prompt_id")}
            appended_prompt_ids: list[str] = []

            for p in prompts:
                c = canonical_prompt_spec(p)
                prompt_id = compute_prompt_id(c)
                if prompt_id in existing_prompt_ids:
                    continue
                rows.append(
                    self._empty_row(
                        prompt_id=prompt_id,
                        prompt=serialize_prompt_content(c),
                    )
                )
                existing_prompt_ids.add(prompt_id)
                appended_prompt_ids.append(prompt_id)

            if appended_prompt_ids:
                self._rewrite_rows(rows)

            self._write_sidecar()

        return appended_prompt_ids

    def retain_only_prompts(self, prompt_ids: set[str]) -> None:
        """Keep only rows whose prompt_id is in prompt_ids; drop the rest. Updates CSV and sidecar."""
        with self._locked_file():
            if not self._csv_path.exists():
                return
            rows = self._read_rows()
            kept = [r for r in rows if r.get(PROMPT_ID_COLUMN) in prompt_ids]
            if len(kept) < len(rows):
                self._rewrite_rows(kept)
            self._write_sidecar()

    def write_cell(self, prompt_id: str, alias: str, cell: MatrixCell) -> None:
        if alias not in self._model_aliases:
            raise KeyError(f"Unknown model alias: {alias}")

        with self._locked_file():
            rows = self._read_rows()
            target_row = self._locate_row(rows=rows, prompt_id=prompt_id, alias=alias)
            target_row[alias] = cell.to_csv_cell()
            self._rewrite_rows(rows)

    def _empty_row(self, *, prompt_id: str, prompt: str) -> dict[str, str]:
        pending_cell = MatrixCell(status=CellStatus.PENDING).to_csv_cell()
        return {
            PROMPT_ID_COLUMN: prompt_id,
            PROMPT_COLUMN: prompt,
            **dict.fromkeys(self._model_aliases, pending_cell),
        }

    def _read_rows(self) -> list[dict[str, str]]:
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Matrix CSV does not exist: {self._csv_path}")

        with self._csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file, **csv_writer_kwargs())
            fieldnames = list(reader.fieldnames or [])
            if fieldnames != self._headers:
                raise ValueError(f"Unexpected matrix CSV headers in {self._csv_path}")
            return [
                {header: row.get(header, MISSING_CELL) for header in self._headers}
                for row in reader
            ]

    def _locate_row(
        self, *, rows: list[dict[str, str]], prompt_id: str, alias: str
    ) -> dict[str, str]:
        pid = prompt_id.strip() if isinstance(prompt_id, str) else ""
        matching_rows = [row for row in rows if (row.get(PROMPT_ID_COLUMN) or "").strip() == pid]
        if not matching_rows:
            raise KeyError(f"Unknown prompt_id: {prompt_id!r}")

        for row in matching_rows:
            if row[alias] == MISSING_CELL:
                return row

        if len(matching_rows) == 1:
            return matching_rows[0]

        raise ValueError(f"Ambiguous completed cell for prompt_id={prompt_id!r} alias={alias!r}")

    def _rewrite_rows(self, rows: list[dict[str, str]]) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=self._csv_path.parent,
            prefix=f".{self._csv_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(temp_file, fieldnames=self._headers, **csv_writer_kwargs())
            writer.writeheader()
            writer.writerows(rows)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, self._csv_path)

    def _write_sidecar(self) -> None:
        sidecar_payload = build_sidecar_metadata(model_aliases=self._model_aliases)
        sidecar_payload["schema_version"] = SCHEMA_VERSION
        sidecar_path = metadata_sidecar_path(self._csv_path)
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @contextmanager
    def _locked_file(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock_file:
            _acquire_lock(lock_file)
            try:
                yield
            finally:
                _release_lock(lock_file)


if os.name == "nt":
    import msvcrt

    def _acquire_lock(lock_file: BinaryIO) -> None:
        lock_file.seek(0)
        if lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]

    def _release_lock(lock_file: BinaryIO) -> None:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

else:
    import fcntl

    def _acquire_lock(lock_file: BinaryIO) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    def _release_lock(lock_file: BinaryIO) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = ["MatrixCSVWriter", "build_experiment_csv_path", "load_existing_matrix"]
