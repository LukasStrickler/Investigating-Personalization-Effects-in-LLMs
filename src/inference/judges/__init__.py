"""LLM-as-a-judge package — resumable, judge-parallel, sentinel-parsed.

Public API:

    from inference.judges import (
        JudgeConfig, JudgeExecutionConfig, JudgeSubject,
        JudgeRunner, run_judges, JudgeResult, JudgeStatus, JudgeVerdict,
        ExperimentDataFrameAdapter, GenericRecordsAdapter,
        judge_config_hash,
    )

These names are also re-exported lazily from ``inference``.
"""

from __future__ import annotations

from inference.judges.adapters import (
    ExperimentDataFrameAdapter,
    GenericRecordsAdapter,
    SubjectAdapter,
    subjects_from_dataframe,
)
from inference.judges.log import JudgeLogger
from inference.judges.parsing import ParseOutcome, parse_final_answer
from inference.judges.prompts import (
    build_judge_messages,
    render_transcript,
)
from inference.judges.runner import JudgeRunner, run_judges
from inference.judges.types import (
    JUDGE_PROMPT_VERSION,
    NONE_SENTINEL,
    JudgeConfig,
    JudgeExecutionConfig,
    JudgeResult,
    JudgeStatus,
    JudgeSubject,
    JudgeVerdict,
    LogVerbosity,
    ParseStatus,
    judge_config_hash,
)

__all__ = [
    "ExperimentDataFrameAdapter",
    "GenericRecordsAdapter",
    "JUDGE_PROMPT_VERSION",
    "JudgeConfig",
    "JudgeExecutionConfig",
    "JudgeLogger",
    "JudgeResult",
    "JudgeRunner",
    "JudgeStatus",
    "JudgeSubject",
    "JudgeVerdict",
    "LogVerbosity",
    "NONE_SENTINEL",
    "ParseOutcome",
    "ParseStatus",
    "SubjectAdapter",
    "build_judge_messages",
    "judge_config_hash",
    "parse_final_answer",
    "render_transcript",
    "run_judges",
    "subjects_from_dataframe",
]
