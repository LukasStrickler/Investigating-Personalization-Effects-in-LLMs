from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from inference.experiments.csv_schema import (
    PROMPT_COLUMN,
    PROMPT_ID_COLUMN,
    CellStatus,
    MatrixCell,
    csv_writer_kwargs,
)


def build_dataframe_from_csv(csv_path: Path) -> pd.DataFrame:
    """Build raw experiment DataFrame: prompt_id, prompt, then one column per model. Cells are dicts (status, response, error_message, metadata)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Matrix CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, **csv_writer_kwargs())
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"Matrix CSV is missing headers: {csv_path}")
        if headers[0] != PROMPT_ID_COLUMN:
            raise ValueError(f"Matrix CSV must start with '{PROMPT_ID_COLUMN}': {csv_path}")
        if PROMPT_COLUMN not in headers:
            raise ValueError(f"Matrix CSV must have '{PROMPT_COLUMN}' column (second column).")

        aliases = [h for h in headers if h not in (PROMPT_ID_COLUMN, PROMPT_COLUMN)]
        rows = [_build_raw_row(raw_row, aliases) for raw_row in reader]

    return pd.DataFrame(rows, columns=[PROMPT_ID_COLUMN, PROMPT_COLUMN, *aliases])


def _build_raw_row(raw_row: dict[str, str | None], aliases: list[str]) -> dict[str, Any]:
    prompt_id = raw_row.get(PROMPT_ID_COLUMN)
    if prompt_id is None or str(prompt_id).strip() == "":
        raise ValueError("Matrix CSV row is missing prompt_id.")
    prompt_id = str(prompt_id).strip()
    prompt_raw = raw_row.get(PROMPT_COLUMN) or ""

    row: dict[str, Any] = {PROMPT_ID_COLUMN: prompt_id, PROMPT_COLUMN: prompt_raw}
    for alias in aliases:
        try:
            cell = MatrixCell.from_csv_cell(str(raw_row.get(alias, "") or ""))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Malformed matrix cell for prompt_id={prompt_id!r}, alias={alias!r}."
            ) from error
        row[alias] = _cell_to_dict(cell)
    return row


def _cell_to_dict(cell: MatrixCell | None) -> dict[str, Any]:
    if cell is None:
        return {"status": None, "response": None, "error_message": None, "metadata": None}
    return {
        "status": cell.status.value,
        "response": cell.response,
        "error_message": cell.error_message,
        "metadata": cell.metadata,
    }


def filter_experiment_dataframe(
    raw_df: pd.DataFrame,
    *,
    models: list[str] | None = None,
    all_complete: bool = False,
    min_success_per_row: int | None = None,
    prompt_contains: str | None = None,
    status: CellStatus | str | None = None,
) -> pd.DataFrame:
    """Filter raw experiment DataFrame; returns same schema, subset of rows/columns.

    - models: restrict to these model columns (default: all).
    - all_complete: keep only rows where every selected model cell has status success.
    - min_success_per_row: keep only rows with at least this many success cells in selected models.
    - prompt_contains: keep only rows where the full prompt column contains this substring.
    - status: keep only rows where every selected model cell has this status.
    """
    if PROMPT_ID_COLUMN not in raw_df.columns or PROMPT_COLUMN not in raw_df.columns:
        raise ValueError("DataFrame must have prompt_id and prompt columns (universal raw shape).")
    model_cols = [c for c in raw_df.columns if c not in (PROMPT_ID_COLUMN, PROMPT_COLUMN)]
    if not model_cols:
        return raw_df.copy()

    if models is not None:
        missing = [m for m in models if m not in raw_df.columns]
        if missing:
            raise ValueError(f"Model columns not in DataFrame: {missing}")
        model_cols = [m for m in models if m in raw_df.columns]
    cols = [PROMPT_ID_COLUMN, PROMPT_COLUMN, *model_cols]
    out = raw_df[cols].copy()

    def _row_prompt(row: pd.Series) -> str:
        return str(row.get(PROMPT_COLUMN, "") or "")

    mask = pd.Series(True, index=out.index)
    if prompt_contains is not None:
        sub = prompt_contains
        mask &= out.apply(lambda row: sub in _row_prompt(row), axis=1)

    if all_complete or min_success_per_row is not None or status is not None:
        status_val = status.value if isinstance(status, CellStatus) else status

        def _row_ok(row: pd.Series) -> bool:
            cells = [row.get(c) for c in model_cols]
            statuses = []
            for c in cells:
                if isinstance(c, dict):
                    statuses.append(c.get("status"))
                else:
                    statuses.append(None)
            if status is not None and not all(s == status_val for s in statuses):
                return False
            if all_complete and not all(s == CellStatus.SUCCESS.value for s in statuses):
                return False
            if min_success_per_row is not None:
                success_count = sum(1 for s in statuses if s == CellStatus.SUCCESS.value)
                if success_count < min_success_per_row:
                    return False
            return True

        mask &= out.apply(_row_ok, axis=1)

    return out.loc[mask].reset_index(drop=True)


def to_analysis_dataframe(raw_df: pd.DataFrame, **filter_kwargs: Any) -> pd.DataFrame:
    """Transform raw experiment DataFrame to analysis shape: prompt_id, prompt (full raw), then per-model response text.

    Prompt is the raw prompt column as stored (canonical JSON or string). Model columns are response text or None
    when the cell did not succeed. If filter_kwargs are provided, filter_experiment_dataframe is applied first.
    """
    if filter_kwargs:
        raw_df = filter_experiment_dataframe(raw_df, **filter_kwargs)
    if PROMPT_ID_COLUMN not in raw_df.columns or PROMPT_COLUMN not in raw_df.columns:
        raise ValueError("DataFrame must have prompt_id and prompt columns (universal raw shape).")
    model_cols = [c for c in raw_df.columns if c not in (PROMPT_ID_COLUMN, PROMPT_COLUMN)]

    rows = []
    for _, row in raw_df.iterrows():
        prompt_id = row[PROMPT_ID_COLUMN]
        prompt_raw = str(row.get(PROMPT_COLUMN) or "")
        out_row: dict[str, Any] = {
            PROMPT_ID_COLUMN: prompt_id,
            PROMPT_COLUMN: prompt_raw,
        }
        for alias in model_cols:
            cell = row.get(alias)
            if isinstance(cell, dict) and cell.get("status") == CellStatus.SUCCESS.value:
                out_row[alias] = cell.get("response")
            else:
                out_row[alias] = None
        rows.append(out_row)

    return pd.DataFrame(rows, columns=[PROMPT_ID_COLUMN, PROMPT_COLUMN, *model_cols])


__all__ = [
    "build_dataframe_from_csv",
    "filter_experiment_dataframe",
    "to_analysis_dataframe",
]
