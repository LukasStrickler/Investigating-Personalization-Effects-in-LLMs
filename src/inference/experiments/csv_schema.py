"""Experiment CSV schema: cell payload, identities, and canonical serialization.

Traceability (stable for production):
- prompt_id = sha256(canonical_json(prompt)); row key; one row per prompt combination.
- prompt column = canonical serialized prompt per row; cells do not duplicate it.
- cell identity = (prompt_id, model_alias); cell payload = {status, response?, error_message?, metadata?}.
"""

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
"""Column name for the row key: sha256(canonical_json(prompt)). First column in CSV."""
PROMPT_COLUMN = "prompt"
"""Column name for the canonical serialized prompt per row. Second column."""
SCHEMA_VERSION = 2
MISSING_CELL = ""


class CellStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    NOT_REQUESTED = "not_requested"  # not scheduled (e.g. sparse grid)
    PENDING = "pending"
    RETRYING = "retrying"


TERMINAL_CELL_STATUSES = frozenset(
    {
        CellStatus.SUCCESS,
        CellStatus.FAILED,
        CellStatus.RATE_LIMITED,
        CellStatus.NOT_REQUESTED,
    }
)
RETRYABLE_CELL_STATUSES = frozenset({CellStatus.FAILED, CellStatus.RATE_LIMITED})


@dataclass(frozen=True, slots=True)
class MatrixCell:
    status: CellStatus
    response: JSONValue | None = None
    error_message: str | None = None
    """Optional metadata (e.g. system_prompt_folded + system_prompt when provider folded system into user)."""
    metadata: dict[str, Any] | None = None

    def to_csv_cell(self) -> str:
        payload: dict[str, Any] = {"status": self.status.value}
        if self.response is not None:
            payload["response"] = self.response
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        if self.metadata is not None and len(self.metadata) > 0:
            payload["metadata"] = self.metadata
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

        try:
            status_enum = CellStatus(raw_status)
        except ValueError:
            valid = ", ".join(s.value for s in CellStatus)
            raise ValueError(
                f"Matrix cell has invalid status {raw_status!r}. Valid: {valid}"
            ) from None

        meta = payload.get("metadata")
        if meta is not None and not isinstance(meta, dict):
            meta = None

        return cls(
            status=status_enum,
            response=payload.get("response"),
            error_message=payload.get("error_message"),
            metadata=meta,
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
    """Headers: prompt_id, prompt, then one column per model alias."""
    aliases = _validated_aliases(model_aliases)
    return [PROMPT_ID_COLUMN, PROMPT_COLUMN, *aliases]


def canonical_prompt_spec(spec: str | dict[str, Any]) -> dict[str, Any]:
    """Normalize any prompt spec to a single canonical form: {"messages": [{"role": str, "content": str}, ...]}.

    Ensures consistency: single text prompts become one user message; system+user become system then user.
    Use this before compute_prompt_id and serialize_prompt_content so stored prompts have one JSON shape.
    """
    if isinstance(spec, str):
        s = spec.strip()
        return {"messages": [{"role": "user", "content": s}]}
    if not isinstance(spec, dict):
        raise TypeError("Prompt spec must be str or dict.")
    messages: list[dict[str, str]] = []
    if "messages" in spec and isinstance(spec.get("messages"), list):
        msgs = list(spec["messages"])
        if spec.get("system") not in (None, ""):
            messages.append({"role": "system", "content": str(spec["system"])})
        messages.extend(msgs)
    else:
        if spec.get("system") not in (None, ""):
            messages.append({"role": "system", "content": str(spec["system"])})
        user = spec.get("user")
        if user not in (None, ""):
            messages.append({"role": "user", "content": str(user)})
    if not messages:
        raise ValueError("Prompt spec has no system or user content.")
    return {"messages": messages}


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
            "optional_keys": ["response", "error_message", "metadata"],
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
    "PROMPT_COLUMN",
    "PROMPT_ID_COLUMN",
    "RETRYABLE_CELL_STATUSES",
    "SCHEMA_VERSION",
    "TERMINAL_CELL_STATUSES",
    "build_matrix_headers",
    "build_sidecar_metadata",
    "canonical_prompt_spec",
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
