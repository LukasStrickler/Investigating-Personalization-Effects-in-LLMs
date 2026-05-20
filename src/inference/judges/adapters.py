"""Input adapters: experiment dataframe/CSV and generic records.

Note on coupling: ExperimentDataFrameAdapter consumes the *public* envelope of
experiment matrix cells (``{"status": ..., "response": ...}``) defined in
``inference.experiments.csv_schema``. If that envelope changes, update this file.
We intentionally do not import experiments code; the parsing is by string convention
so the judges package stays decoupled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

from inference.judges.types import JudgeSubject

_EXPERIMENT_METADATA_COLS = ("prompt_id", "prompt")


class SubjectAdapter(Protocol):
    def iter_subjects(self) -> Iterator[JudgeSubject]: ...
    def summary(self) -> dict[str, Any]: ...


def _is_missing_scalar(value: Any) -> bool:
    """True for pandas NA/NaN and other missing scalars."""
    if value is None:
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except ImportError:
        pass
    if isinstance(value, float) and value != value:  # NaN
        return True
    return False


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class GenericRecordsAdapter:
    """Adapt a list of dicts (or a pandas DataFrame) into JudgeSubjects."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]] | Any,
        *,
        id_field: str | None = None,
        content_field: str | None = "content",
        messages_field: str | None = None,
        subject_model_alias_field: str | None = None,
        metadata_fields: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        self._records = self._normalize(records)
        self._id_field = id_field
        self._content_field = content_field
        self._messages_field = messages_field
        self._model_field = subject_model_alias_field
        self._metadata_fields = metadata_fields
        self._source_id = source_id
        self._counts = {"total": len(self._records), "emitted": 0, "skipped_empty": 0}

    @staticmethod
    def _normalize(records: Any) -> list[dict[str, Any]]:
        if hasattr(records, "to_dict") and callable(records.to_dict):
            return list(records.to_dict(orient="records"))
        return [dict(r) for r in records]

    def iter_subjects(self) -> Iterator[JudgeSubject]:
        for i, rec in enumerate(self._records):
            messages = rec.get(self._messages_field) if self._messages_field else None
            content = rec.get(self._content_field) if self._content_field else None
            if self._messages_field:
                if not isinstance(messages, list) or len(messages) == 0:
                    self._counts["skipped_empty"] += 1
                    continue
            elif content is None or content == "":
                self._counts["skipped_empty"] += 1
                continue
            sid = (
                str(rec[self._id_field])
                if self._id_field and rec.get(self._id_field) is not None
                else _stable_hash({"i": i, "rec": rec})
            )
            md: dict[str, Any] = {}
            if self._metadata_fields:
                for f in self._metadata_fields:
                    if f in rec:
                        md[f] = rec[f]
            self._counts["emitted"] += 1
            yield JudgeSubject(
                subject_id=sid,
                subject_content=content if isinstance(content, str) else None,
                messages=list(messages) if isinstance(messages, list) else None,
                subject_model_alias=(
                    str(rec[self._model_field])
                    if self._model_field and rec.get(self._model_field) is not None
                    else None
                ),
                source_id=self._source_id,
                prompt_id=None,
                metadata=md or None,
            )

    def summary(self) -> dict[str, Any]:
        return dict(self._counts)


class ExperimentDataFrameAdapter:
    """Adapt an experiment raw matrix (DataFrame or CSV path) into JudgeSubjects.

    Each successful model cell becomes one JudgeSubject:
      - subject_id      = prompt_id from the matrix
      - subject_model_alias = column header (model alias)
      - subject_content = the cell's "response" field
      - prompt_id       = same as subject_id (lineage)
      - metadata        = {"prompt_spec": <prompt spec, if column present>}
    """

    def __init__(
        self,
        source: Any,
        *,
        only_models: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        self._df = self._load_df(source)
        self._only_models = set(only_models) if only_models else None
        self._source_id = source_id if source_id is not None else (
            str(source) if isinstance(source, (str, Path)) else None
        )
        self._counts: dict[str, int] = {
            "rows": 0,
            "subjects_emitted": 0,
            "skipped_non_success": 0,
            "skipped_unparseable": 0,
        }

    @staticmethod
    def _load_df(source: Any) -> Any:
        import pandas as pd

        if isinstance(source, (str, Path)):
            return pd.read_csv(source)
        # Assume a DataFrame
        return source

    def _model_columns(self) -> list[str]:
        cols = [c for c in self._df.columns if c not in _EXPERIMENT_METADATA_COLS]
        if self._only_models is not None:
            cols = [c for c in cols if c in self._only_models]
        return cols

    def iter_subjects(self) -> Iterator[JudgeSubject]:
        cols = self._model_columns()
        for _, row in self._df.iterrows():
            self._counts["rows"] += 1
            raw_pid = row.get("prompt_id")
            if raw_pid is None or _is_missing_scalar(raw_pid):
                continue
            prompt_id = str(raw_pid).strip()
            if not prompt_id or prompt_id.casefold() == "nan":
                continue
            prompt_spec = row.get("prompt") if "prompt" in self._df.columns else None
            for alias in cols:
                cell_raw = row.get(alias)
                cell = self._parse_cell(cell_raw)
                if cell is None:
                    self._counts["skipped_unparseable"] += 1
                    continue
                if cell.get("status") != "success":
                    self._counts["skipped_non_success"] += 1
                    continue
                response = cell.get("response") or ""
                if not isinstance(response, str) or response == "":
                    self._counts["skipped_non_success"] += 1
                    continue
                self._counts["subjects_emitted"] += 1
                yield JudgeSubject(
                    subject_id=prompt_id,
                    subject_content=response,
                    messages=None,
                    subject_model_alias=alias,
                    source_id=self._source_id,
                    prompt_id=prompt_id,
                    metadata={"prompt_spec": prompt_spec} if prompt_spec is not None else None,
                )

    @staticmethod
    def _parse_cell(raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        s = raw.strip()
        if not s or s == "__missing__":
            return None
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def summary(self) -> dict[str, Any]:
        return dict(self._counts)


def subjects_from_dataframe(
    df: Any,
    *,
    id_field: str | None = None,
    content_field: str | None = "content",
    messages_field: str | None = None,
    subject_model_alias_field: str | None = None,
    metadata_fields: list[str] | None = None,
    source_id: str | None = None,
) -> list[JudgeSubject]:
    """Convenience: build a list[JudgeSubject] from a pandas DataFrame.

    Equivalent to ``list(GenericRecordsAdapter(df, ...).iter_subjects())`` but is the
    canonical dataframe-in entry point. Use ``metadata_fields`` to carry analysis
    columns (true labels, group keys, ...) through to the output dataframe.
    """
    adapter = GenericRecordsAdapter(
        df,
        id_field=id_field,
        content_field=content_field,
        messages_field=messages_field,
        subject_model_alias_field=subject_model_alias_field,
        metadata_fields=metadata_fields,
        source_id=source_id,
    )
    return list(adapter.iter_subjects())


def coerce_to_adapter(input_obj: Any) -> SubjectAdapter:
    """Best-effort coercion: lists of JudgeSubject are wrapped; adapters pass through.

    Pandas DataFrames are NOT auto-coerced — column semantics are too ambiguous to
    guess. Use ``subjects_from_dataframe(df, content_field=..., id_field=...)`` or
    construct a ``GenericRecordsAdapter`` explicitly.
    """
    if isinstance(input_obj, list) and (
        not input_obj or isinstance(input_obj[0], JudgeSubject)
    ):
        return _BareSubjectsAdapter(input_obj)
    if hasattr(input_obj, "iter_subjects") and hasattr(input_obj, "summary"):
        return input_obj  # type: ignore[no-any-return]
    raise TypeError(
        "Pass a SubjectAdapter (e.g. ExperimentDataFrameAdapter / GenericRecordsAdapter) "
        "or a list[JudgeSubject]"
    )


class _BareSubjectsAdapter:
    def __init__(self, subjects: list[JudgeSubject]) -> None:
        self._subjects = list(subjects)

    def iter_subjects(self) -> Iterator[JudgeSubject]:
        yield from self._subjects

    def summary(self) -> dict[str, Any]:
        return {"total": len(self._subjects), "emitted": len(self._subjects)}


__all__ = [
    "ExperimentDataFrameAdapter",
    "GenericRecordsAdapter",
    "SubjectAdapter",
    "coerce_to_adapter",
    "subjects_from_dataframe",
]
