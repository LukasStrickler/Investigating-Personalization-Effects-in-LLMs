"""High-level experiment API types for matrix-style inference workflows.

Boundary definition:
- ``inference`` stays the low-level runtime/provider layer and keeps request-level
  concerns such as LiteLLM adapters, retries, rate limits, and logging.
- ``inference.experiments`` is a high-level research workflow layer responsible
  for prompt/model matrix orchestration, CSV resume semantics, and aggregate
  results for analysis.

This module intentionally defines API contracts only. It does not implement the
actual execution behavior for running experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from inference.client import UnifiedInferenceClient


VerbosityLevel = Literal["quiet", "normal", "verbose", "debug"]


@dataclass(frozen=True, slots=True)
class ExperimentRetryOptions:
    """Retry behavior for each cell in the prompt x model matrix.

    Defaults mirror low-level retry policy defaults to preserve behavior unless
    experiments explicitly override them.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be > 0")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be > 0")
        if self.base_delay > self.max_delay:
            raise ValueError("base_delay must be <= max_delay")


@dataclass(frozen=True, slots=True)
class ExperimentSchedulingOptions:
    """Scheduling controls for high-level experiment orchestration."""

    max_concurrency: int = 0
    per_model_concurrency: int = 0
    interleave_model_aliases: bool = True
    max_retry_after_wait_seconds: float = 3600.0

    def __post_init__(self) -> None:
        if self.max_concurrency < 0:
            raise ValueError("max_concurrency must be >= 0")
        if self.per_model_concurrency < 0:
            raise ValueError("per_model_concurrency must be >= 0")
        if self.max_retry_after_wait_seconds <= 0:
            raise ValueError("max_retry_after_wait_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a complete prompt x model experiment run.

    Required inputs:
    - ``experiment_name`` labels outputs and metadata.
    - ``model_aliases`` defines the model columns in the matrix.
    - ``prompts`` defines the row inputs and must contain at least one prompt.

    Resume behavior:
    - ``resume_from_existing_csv`` enables recovery from an existing CSV.
    - ``existing_csv_path`` can be provided explicitly; if omitted, the runner is
      expected to resolve the default experiment CSV path from conventions.
    """

    experiment_name: str
    model_aliases: list[str]
    prompts: list[str]
    retry: ExperimentRetryOptions = field(default_factory=ExperimentRetryOptions)
    scheduling: ExperimentSchedulingOptions = field(default_factory=ExperimentSchedulingOptions)
    verbosity: VerbosityLevel = "normal"
    resume_from_existing_csv: bool = False
    existing_csv_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if not self.model_aliases:
            raise ValueError("model_aliases must contain at least one model alias")
        if not self.prompts:
            raise ValueError("prompts must contain at least one prompt")
        if any(not alias.strip() for alias in self.model_aliases):
            raise ValueError("model_aliases cannot contain empty values")
        if any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("prompts cannot contain empty values")


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Aggregate counters reported after full matrix completion."""

    prompt_count: int
    model_count: int
    total_cells: int
    completed_cells: int
    failed_cells: int
    rate_limited_cells: int


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Completed experiment artifact metadata and tabular output.

    ``dataframe`` always represents the fully materialized matrix state for the
    run. ``csv_path`` and ``csv_name`` identify the durable CSV used as source of
    truth during execution.
    """

    dataframe: Any
    csv_path: Path
    csv_name: str
    summary: ExperimentSummary


class ExperimentRunner:
    """Facade for high-level experiment execution.

    The runner composes low-level ``UnifiedInferenceClient`` calls without
    changing client semantics. It orchestrates the full prompt x model matrix,
    applies experiment-level retry/scheduling options, and handles CSV resume
    behavior.
    """

    def __init__(self, client: UnifiedInferenceClient) -> None:
        self._client = client

    async def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run one experiment and return only after full matrix completion.

        This API is intentionally non-streaming: callers receive a single
        ``ExperimentResult`` after all prompt/model cells are processed and final
        DataFrame + CSV metadata are available.
        """
        from inference.experiments.runner import ExperimentRunner as ConcreteExperimentRunner

        return await ConcreteExperimentRunner(self._client).run(config)


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRetryOptions",
    "ExperimentRunner",
    "ExperimentSchedulingOptions",
    "ExperimentSummary",
    "VerbosityLevel",
]
