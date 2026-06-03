"""Judge CSV schema. One row per (subject_id, subject_model_alias, judge_alias, judge_config_hash)."""

from __future__ import annotations

import json
from typing import Any

from inference.judges.types import JudgeStatus, JudgeVerdict, ParseStatus

SCHEMA_VERSION = 2

COLUMNS: list[str] = [
    "judgment_id",
    "subject_id",
    "source_id",
    "prompt_id",
    "subject_model_alias",
    "judge_alias",
    "judge_config_hash",
    "status",
    "raw_output",
    "final_class",
    "none_declared",
    "parse_status",
    "error_message",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "retry_count",
    "started_at",
    "completed_at",
    "metadata",
]


def csv_writer_kwargs() -> dict[str, Any]:
    return {"quoting": 1, "lineterminator": "\n"}  # csv.QUOTE_ALL


def resume_key(
    subject_id: str,
    subject_model_alias: str | None,
    judge_alias: str,
    judge_config_hash: str,
) -> tuple[str, str, str, str]:
    return (subject_id, subject_model_alias or "", judge_alias, judge_config_hash)


def _json_or_none(s: str) -> Any:
    if s == "" or s is None:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def verdict_to_row(v: JudgeVerdict) -> dict[str, str]:
    return {
        "judgment_id": v.judgment_id,
        "subject_id": v.subject_id,
        "source_id": v.source_id or "",
        "prompt_id": v.prompt_id or "",
        "subject_model_alias": v.subject_model_alias or "",
        "judge_alias": v.judge_alias,
        "judge_config_hash": v.judge_config_hash,
        "status": v.status.value,
        "raw_output": v.raw_output,
        "final_class": v.final_class or "",
        "none_declared": "true" if v.none_declared else "false",
        "parse_status": v.parse_status.value,
        "error_message": v.error_message or "",
        "prompt_tokens": "" if v.prompt_tokens is None else str(v.prompt_tokens),
        "completion_tokens": "" if v.completion_tokens is None else str(v.completion_tokens),
        "total_tokens": "" if v.total_tokens is None else str(v.total_tokens),
        "latency_ms": f"{v.latency_ms:.3f}",
        "retry_count": str(v.retry_count),
        "started_at": v.started_at,
        "completed_at": v.completed_at,
        "metadata": json.dumps(v.metadata, sort_keys=True, ensure_ascii=False) if v.metadata else "",
    }


def row_to_verdict(row: dict[str, str]) -> JudgeVerdict:
    def _int(s: str) -> int | None:
        return int(s) if s not in ("", None) else None

    def _float(s: str) -> float:
        return float(s) if s not in ("", None) else 0.0

    return JudgeVerdict(
        judgment_id=row["judgment_id"],
        subject_id=row["subject_id"],
        source_id=row.get("source_id") or None,
        prompt_id=row.get("prompt_id") or None,
        subject_model_alias=row.get("subject_model_alias") or None,
        judge_alias=row["judge_alias"],
        judge_config_hash=row["judge_config_hash"],
        status=JudgeStatus(row["status"]),
        raw_output=row.get("raw_output", ""),
        final_class=row.get("final_class") or None,
        none_declared=row.get("none_declared", "false") == "true",
        parse_status=ParseStatus(row.get("parse_status", ParseStatus.MISSING_SENTINEL.value)),
        error_message=row.get("error_message") or None,
        prompt_tokens=_int(row.get("prompt_tokens", "")),
        completion_tokens=_int(row.get("completion_tokens", "")),
        total_tokens=_int(row.get("total_tokens", "")),
        latency_ms=_float(row.get("latency_ms", "0")),
        retry_count=int(row.get("retry_count") or 0),
        started_at=row.get("started_at", ""),
        completed_at=row.get("completed_at", ""),
        metadata=_json_or_none(row.get("metadata", "")) or {},
    )


__all__ = [
    "COLUMNS",
    "SCHEMA_VERSION",
    "csv_writer_kwargs",
    "resume_key",
    "row_to_verdict",
    "verdict_to_row",
]
