from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from inference.experiments.csv_schema import (
    PROMPT_ID_COLUMN,
    MatrixCell,
    csv_writer_kwargs,
    metadata_sidecar_path,
)


def build_dataframe_from_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Matrix CSV does not exist: {csv_path}")

    prompt_text_by_id = _load_prompt_text_by_id(csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, **csv_writer_kwargs())
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"Matrix CSV is missing headers: {csv_path}")
        if headers[0] != PROMPT_ID_COLUMN:
            raise ValueError(f"Matrix CSV must start with '{PROMPT_ID_COLUMN}': {csv_path}")

        aliases = headers[1:]
        rows = [
            _build_dataframe_row(
                raw_row=raw_row, aliases=aliases, prompt_text_by_id=prompt_text_by_id
            )
            for raw_row in reader
        ]

    return pd.DataFrame(rows, columns=[PROMPT_ID_COLUMN, "prompt", *aliases])


def _build_dataframe_row(
    *, raw_row: dict[str, str | None], aliases: list[str], prompt_text_by_id: dict[str, str]
) -> dict[str, Any]:
    prompt_id = raw_row.get(PROMPT_ID_COLUMN)
    if prompt_id is None or prompt_id.strip() == "":
        raise ValueError("Matrix CSV row is missing prompt_id.")

    prompt = prompt_text_by_id.get(prompt_id)
    if prompt is None:
        raise ValueError(f"Prompt text metadata missing for prompt_id={prompt_id!r}.")

    row: dict[str, Any] = {PROMPT_ID_COLUMN: prompt_id, "prompt": prompt}
    for alias in aliases:
        try:
            cell = MatrixCell.from_csv_cell(raw_row.get(alias, "") or "")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Malformed matrix cell for prompt_id={prompt_id!r}, alias={alias!r}."
            ) from error

        row[alias] = {
            "status": cell.status.value if cell is not None else None,
            "response": None if cell is None else cell.response,
            "error_message": None if cell is None else cell.error_message,
        }
    return row


def _load_prompt_text_by_id(csv_path: Path) -> dict[str, str]:
    sidecar_path = metadata_sidecar_path(csv_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Matrix metadata sidecar does not exist: {sidecar_path}")

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Matrix metadata sidecar is not valid JSON: {sidecar_path}") from error

    prompt_text_by_id = payload.get("prompt_text_by_id")
    if not isinstance(prompt_text_by_id, dict):
        raise ValueError(f"Matrix metadata sidecar is missing prompt_text_by_id: {sidecar_path}")

    normalized: dict[str, str] = {}
    for prompt_id, prompt in prompt_text_by_id.items():
        if not isinstance(prompt_id, str) or not isinstance(prompt, str):
            raise ValueError(f"Matrix metadata sidecar has invalid prompt mapping: {sidecar_path}")
        normalized[prompt_id] = prompt
    return normalized


__all__ = ["build_dataframe_from_csv"]
