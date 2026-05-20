"""Public type contracts for the LLM-as-a-judge package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

LogVerbosity = Literal["silent", "normal", "verbose", "debug"]


class JudgeStatus(str, Enum):
    SUCCESS = "success"
    CLASSIFICATION_FAILED = "classification_failed"
    CALL_FAILED = "call_failed"
    NOT_REQUESTED = "not_requested"


class ParseStatus(str, Enum):
    MATCHED = "matched"
    NONE_DECLARED = "none_declared"
    UNMATCHED = "unmatched"
    MISSING_SENTINEL = "missing_sentinel"
    FREE_FORM = "free_form"


# Bumped when the judge prompt scaffold changes; participates in judge_config_hash.
JUDGE_PROMPT_VERSION = 2
# Reserved literal the judge can emit inside <final_answer> when no class applies.
NONE_SENTINEL = "__NONE__"


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    experiment_name: str
    judges: list[str]
    judge_prompt: str
    classes: list[str] | None = None
    max_tokens: int | None = 512
    temperature: float = 0.0
    thinking_budget_tokens: int | None = None
    include_metadata_in_prompt: bool = False
    output_dir: Path | None = None
    resume: bool = True
    log_verbosity: LogVerbosity = "normal"

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if not self.judges:
            raise ValueError("judges must contain at least one alias")
        if any(not a.strip() for a in self.judges):
            raise ValueError("judges cannot contain empty aliases")
        if not self.judge_prompt.strip():
            raise ValueError("judge_prompt must be non-empty")
        if self.thinking_budget_tokens is not None and self.thinking_budget_tokens <= 0:
            raise ValueError("thinking_budget_tokens must be a positive integer when set")
        if self.classes is not None:
            if not self.classes:
                raise ValueError("classes, if set, must be non-empty (use None to disable)")
            seen: set[str] = set()
            for c in self.classes:
                if not isinstance(c, str):
                    raise ValueError(f"classes must be strings; got {type(c).__name__}")
                if not c.strip():
                    raise ValueError("class labels cannot be blank")
                if c in seen:
                    raise ValueError(f"duplicate class label: {c!r}")
                if c == NONE_SENTINEL:
                    raise ValueError(f"class label cannot be the reserved sentinel {NONE_SENTINEL!r}")
                seen.add(c)


@dataclass(frozen=True, slots=True)
class JudgeExecutionConfig:
    call_timeout_s: float = 120.0
    per_judge_workers: dict[str, int] | None = None
    default_workers: int = 1

    def workers_for(self, judge_alias: str) -> int:
        if self.per_judge_workers and judge_alias in self.per_judge_workers:
            return max(1, int(self.per_judge_workers[judge_alias]))
        return max(1, int(self.default_workers))


@dataclass(frozen=True, slots=True)
class JudgeSubject:
    subject_id: str
    subject_content: str | None = None
    messages: list[dict[str, Any]] | None = None
    subject_model_alias: str | None = None
    source_id: str | None = None
    prompt_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must be non-empty")
        if self.subject_content is None and not self.messages:
            raise ValueError(
                "JudgeSubject needs either subject_content or messages (non-empty)"
            )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    judgment_id: str
    subject_id: str
    source_id: str | None
    prompt_id: str | None
    subject_model_alias: str | None
    judge_alias: str
    judge_config_hash: str
    status: JudgeStatus
    raw_output: str
    final_class: str | None
    none_declared: bool
    parse_status: ParseStatus
    error_message: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    retry_count: int
    started_at: str
    completed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JudgeResult:
    verdicts: list[JudgeVerdict]
    dataframe: Any
    csv_path: Path
    summary: dict[str, Any]


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def judge_config_hash(config: JudgeConfig) -> str:
    """Stable hash of *semantic* JudgeConfig fields.

    Excludes output paths, resume flag, concurrency. Includes prompt-template version
    constant so bumping it invalidates prior rows on resume.
    """
    payload = {
        "judges": sorted(config.judges),
        "judge_prompt": config.judge_prompt,
        "classes": config.classes,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "thinking_budget_tokens": config.thinking_budget_tokens,
        "include_metadata_in_prompt": config.include_metadata_in_prompt,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "JUDGE_PROMPT_VERSION",
    "JudgeConfig",
    "JudgeExecutionConfig",
    "JudgeResult",
    "JudgeStatus",
    "JudgeSubject",
    "JudgeVerdict",
    "LogVerbosity",
    "NONE_SENTINEL",
    "ParseStatus",
    "judge_config_hash",
]
