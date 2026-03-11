from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from inference.experiments.csv_schema import (
    MISSING_CELL,
    MatrixCell,
    build_matrix_headers,
    build_sidecar_metadata,
    compute_prompt_id,
    csv_writer_kwargs,
    metadata_sidecar_path,
)


def build_experiment_csv_path(experiment_name: str, base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else Path("logs")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return root / experiment_name / f"{timestamp}.csv"


def load_existing_matrix(
    csv_path: Path, model_aliases: list[str]
) -> tuple[set[str], dict[tuple[str, str], str]]:
    """Load existing CSV state for resume mode.

    Returns:
        - prompt_ids_seen: set of prompt_id values in the CSV
        - completed_cells: dict of (prompt_id, alias) -> response for SUCCESS cells only
    """
    from inference.experiments.csv_schema import CellStatus

    prompt_ids: set[str] = set()
    completed_cells: dict[tuple[str, str], str] = {}

    if not csv_path.exists():
        return prompt_ids, completed_cells

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_id = row.get("prompt_id", "")
            if prompt_id:
                prompt_ids.add(prompt_id)
            for alias in model_aliases:
                cell_value = row.get(alias, "")
                if cell_value.strip():
                    cell = MatrixCell.from_csv_cell(cell_value)
                    if cell and cell.status == CellStatus.SUCCESS:
                        completed_cells[(prompt_id, alias)] = str(cell.response or "")
    return prompt_ids, completed_cells


class MatrixCSVWriter:
    def __init__(self, csv_path: Path, model_aliases: list[str]) -> None:
        self._csv_path = csv_path
        self._model_aliases = [
            header for header in build_matrix_headers(model_aliases) if header != "prompt_id"
        ]
        self._headers = build_matrix_headers(model_aliases)
        self._lock_path = Path(f"{csv_path}.lock")

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def initialize(self, prompts: list[str]) -> None:
        rows = [self._empty_row(prompt_id=compute_prompt_id(prompt)) for prompt in prompts]
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked_file():
            self._rewrite_rows(rows)
            self._write_sidecar(prompts)

    def write_cell(self, prompt_id: str, alias: str, cell: MatrixCell) -> None:
        if alias not in self._model_aliases:
            raise KeyError(f"Unknown model alias: {alias}")

        with self._locked_file():
            rows = self._read_rows()
            target_row = self._locate_row(rows=rows, prompt_id=prompt_id, alias=alias)
            target_row[alias] = cell.to_csv_cell()
            self._rewrite_rows(rows)

    def _empty_row(self, *, prompt_id: str) -> dict[str, str]:
        return {
            header: (prompt_id if header == "prompt_id" else MISSING_CELL)
            for header in self._headers
        }

    def _read_rows(self) -> list[dict[str, str]]:
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Matrix CSV does not exist: {self._csv_path}")

        with self._csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file, **csv_writer_kwargs())
            if reader.fieldnames != self._headers:
                raise ValueError(f"Unexpected matrix CSV headers in {self._csv_path}")
            return [
                {header: row.get(header, MISSING_CELL) for header in self._headers}
                for row in reader
            ]

    def _locate_row(
        self, *, rows: list[dict[str, str]], prompt_id: str, alias: str
    ) -> dict[str, str]:
        matching_rows = [row for row in rows if row["prompt_id"] == prompt_id]
        if not matching_rows:
            raise KeyError(f"Unknown prompt_id: {prompt_id}")

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

    def _write_sidecar(self, prompts: list[str]) -> None:
        sidecar_payload = build_sidecar_metadata(model_aliases=self._model_aliases)
        sidecar_payload["prompt_text_by_id"] = {
            compute_prompt_id(prompt): prompt for prompt in prompts
        }

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
