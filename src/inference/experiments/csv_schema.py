from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]

PROMPT_ID_COLUMN = "prompt_id"
SCHEMA_VERSION = 1
MISSING_CELL = ""


class CellStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    PENDING = "pending"
    RETRYING = "retrying"


TERMINAL_CELL_STATUSES = frozenset({CellStatus.SUCCESS, CellStatus.FAILED, CellStatus.RATE_LIMITED})
RETRYABLE_CELL_STATUSES = frozenset({CellStatus.FAILED, CellStatus.RATE_LIMITED})


@dataclass(frozen=True, slots=True)
class MatrixCell:
    status: CellStatus
    response: JSONValue | None = None
    error_message: str | None = None

    def to_csv_cell(self) -> str:
        payload: dict[str, Any] = {"status": self.status.value}
        if self.response is not None:
            payload["response"] = self.response
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_csv_cell(cls, raw_cell: str) -> MatrixCell | None:
        if raw_cell.strip() == "":
            return None

        payload = json.loads(raw_cell)
        if not isinstance(payload, dict):
            raise ValueError("Matrix cell must be a JSON object.")

        raw_status = payload.get("status")
        if not isinstance(raw_status, str):
            raise ValueError("Matrix cell is missing a string status.")

        return cls(
            status=CellStatus(raw_status),
            response=payload.get("response"),
            error_message=payload.get("error_message"),
        )


def csv_writer_kwargs() -> dict[str, Any]:
    return {
        "delimiter": ",",
        "quotechar": '"',
        "doublequote": True,
        "lineterminator": "\n",
        "quoting": csv.QUOTE_MINIMAL,
    }


def build_matrix_headers(model_aliases: list[str]) -> list[str]:
    aliases = _validated_aliases(model_aliases)
    return [PROMPT_ID_COLUMN, *aliases]


def compute_prompt_id(prompt: JSONValue) -> str:
    canonical_prompt = _canonical_json(prompt)
    return hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()


def compute_cell_id(prompt: JSONValue, model_alias: str) -> str:
    return compute_cell_id_from_prompt_id(compute_prompt_id(prompt), model_alias)


def compute_cell_id_from_prompt_id(prompt_id: str, model_alias: str) -> str:
    token = f"{prompt_id}\x1f{model_alias.strip()}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def serialize_prompt_content(prompt: JSONValue) -> str:
    return _canonical_json(prompt)


def deserialize_prompt_content(serialized_prompt: str) -> JSONValue:
    return _parse_json(serialized_prompt)


def serialize_response_content(response: JSONValue) -> str:
    return _canonical_json(response)


def deserialize_response_content(serialized_response: str) -> JSONValue:
    return _parse_json(serialized_response)


def metadata_sidecar_path(csv_path: str | Path) -> Path:
    return Path(f"{csv_path}.meta.json")


def build_sidecar_metadata(*, model_aliases: list[str]) -> dict[str, Any]:
    aliases = _validated_aliases(model_aliases)
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_id_column": PROMPT_ID_COLUMN,
        "model_aliases": aliases,
        "cell_status_values": [status.value for status in CellStatus],
        "missing_cell": MISSING_CELL,
        "prompt_identity": "sha256(canonical_json(prompt))",
        "cell_identity": "sha256(prompt_id + '\\x1f' + alias)",
        "cell_payload": {
            "format": "json",
            "required_key": "status",
            "optional_keys": ["response", "error_message"],
        },
    }


def _canonical_json(content: JSONValue) -> str:
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _parse_json(serialized: str) -> JSONValue:
    value = cast(JSONValue, json.loads(serialized))
    return value


def _validated_aliases(model_aliases: list[str]) -> list[str]:
    aliases = [alias.strip() for alias in model_aliases]
    if not aliases:
        raise ValueError("At least one model alias is required.")
    if any(alias == "" for alias in aliases):
        raise ValueError("Model aliases must be non-empty strings.")
    if len(set(aliases)) != len(aliases):
        raise ValueError("Model aliases must be unique.")
    return aliases


__all__ = [
    "CellStatus",
    "JSONValue",
    "MISSING_CELL",
    "MatrixCell",
    "PROMPT_ID_COLUMN",
    "RETRYABLE_CELL_STATUSES",
    "SCHEMA_VERSION",
    "TERMINAL_CELL_STATUSES",
    "build_matrix_headers",
    "build_sidecar_metadata",
    "compute_cell_id",
    "compute_cell_id_from_prompt_id",
    "compute_prompt_id",
    "csv_writer_kwargs",
    "deserialize_prompt_content",
    "deserialize_response_content",
    "metadata_sidecar_path",
    "serialize_prompt_content",
    "serialize_response_content",
]
