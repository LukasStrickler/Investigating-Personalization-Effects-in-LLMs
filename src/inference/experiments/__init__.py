"""Public API for high-level experiment workflows.

The low-level ``inference`` package remains the runtime/provider layer.
This package exposes only experiment-facing contracts for matrix execution,
resume-from-CSV behavior, and aggregate experiment outputs.

Experiment grid contract (universal raw shape):
- Row = one prompt combination; columns = prompt_id, prompt (canonical serialized), then one per model alias.
- Cell = {status, response?, error_message?, metadata?}; no prompt in cell; status may be not_requested for sparse grids.
- Load CSV with build_dataframe_from_csv → raw DataFrame. Use filter_experiment_dataframe to subset; use to_analysis_dataframe to get prompt (full raw) and response text per model for analysis.

Traceability: prompt_id = sha256(canonical_json(prompt)); cell identity = (prompt_id, model_alias). CSV is UTF-8; single writer per file (locking). Schema version in sidecar.
"""

from __future__ import annotations

from inference.experiments.dataframe import (
    build_dataframe_from_csv,
    filter_experiment_dataframe,
    to_analysis_dataframe,
)
from inference.experiments.prompts import build_experiment_grid
from inference.experiments.runner import ExperimentRunner
from inference.experiments.types import (
    ExperimentConfig,
    ExperimentGrid,
    ExperimentResult,
    ExperimentRetryOptions,
    ExperimentSchedulingOptions,
    ExperimentSummary,
    PromptSpec,
    VerbosityLevel,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentGrid",
    "ExperimentResult",
    "ExperimentRetryOptions",
    "ExperimentRunner",
    "ExperimentSchedulingOptions",
    "ExperimentSummary",
    "PromptSpec",
    "VerbosityLevel",
    "build_dataframe_from_csv",
    "build_experiment_grid",
    "filter_experiment_dataframe",
    "to_analysis_dataframe",
]
