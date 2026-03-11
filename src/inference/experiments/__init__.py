"""Public API for high-level experiment workflows.

The low-level ``inference`` package remains the runtime/provider layer.
This package exposes only experiment-facing contracts for matrix execution,
resume-from-CSV behavior, and aggregate experiment outputs.
"""

from __future__ import annotations

from inference.experiments.dataframe import build_dataframe_from_csv
from inference.experiments.runner import ExperimentRunner
from inference.experiments.types import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRetryOptions,
    ExperimentSchedulingOptions,
    ExperimentSummary,
    VerbosityLevel,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRetryOptions",
    "ExperimentRunner",
    "ExperimentSchedulingOptions",
    "ExperimentSummary",
    "VerbosityLevel",
    "build_dataframe_from_csv",
]
