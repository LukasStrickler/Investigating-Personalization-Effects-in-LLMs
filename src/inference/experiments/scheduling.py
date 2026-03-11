"""Scheduling, retry, and completion semantics for experiment matrices.

An experiment matrix is the Cartesian product of prompts (rows) and model aliases
(columns). This module defines the policy primitives used by experiment runners to
keep retry behavior explicit and matrix completion unambiguous.

Key semantics:
- Normal retryable failures use bounded exponential backoff with a default of
  three retries.
- Provider-declared ``retry-after`` waits are handled separately from normal
  retry exhaustion.
- ``retry-after`` values above one hour are terminal and recorded as
  ``rate_limited`` instead of waiting indefinitely.
- ``await all`` means full matrix completion (all N x M cells terminal), not
  provider-level or column-level completion.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum

from inference.retry import (
    ErrorCategory,
    RetryDecision,
    RetryPolicy,
    calculate_backoff,
    classify_error,
)


class ExperimentCellStatus(str, Enum):
    """Runtime state for a single experiment matrix cell."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` when the cell has reached a terminal state."""
        return self in {
            ExperimentCellStatus.SUCCESS,
            ExperimentCellStatus.FAILED,
            ExperimentCellStatus.RATE_LIMITED,
        }


@dataclass(frozen=True, slots=True)
class ExperimentRetryConfig:
    """Experiment-level retry controls.

    Attributes:
        max_retries: Maximum retries for normal retryable failures.
        max_retry_after_window_seconds: Largest provider-declared ``retry-after``
            wait to honor. Values above this threshold are treated as terminal
            ``rate_limited`` outcomes.
    """

    max_retries: int = 3
    max_retry_after_window_seconds: float = 3600.0

    def __post_init__(self) -> None:
        """Validate config values and keep semantics explicit."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.max_retry_after_window_seconds <= 0:
            raise ValueError("max_retry_after_window_seconds must be > 0")

    def to_retry_policy(self, base_policy: RetryPolicy | None = None) -> RetryPolicy:
        """Build a retry policy honoring experiment-level retry overrides.

        The experiment config owns ``max_retries`` so runs can override provider or
        global defaults without modifying lower-level inference primitives.
        """
        policy = base_policy or RetryPolicy()
        return RetryPolicy(
            max_retries=self.max_retries,
            base_delay=policy.base_delay,
            max_delay=policy.max_delay,
            exponential_base=policy.exponential_base,
            jitter=policy.jitter,
            seed=policy.seed,
        )


class RetryAction(str, Enum):
    """Action to take after a failed inference attempt."""

    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RETRY_AFTER_HINT = "retry_after_hint"
    FAIL = "fail"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True, slots=True)
class RetryResolution:
    """Resolved retry handling for an attempt.

    Attributes:
        action: Next step in the retry flow.
        category: Error category used for observability and reporting.
        wait_seconds: Wait before the next retry for non-terminal actions.
        terminal_status: Terminal cell status for terminal actions.
    """

    action: RetryAction
    category: ErrorCategory
    wait_seconds: float = 0.0
    terminal_status: ExperimentCellStatus | None = None


def resolve_retry(
    *,
    attempt: int,
    error: Exception,
    provider: str | None = None,
    retry_policy: RetryPolicy | None = None,
    config: ExperimentRetryConfig | None = None,
    provider_retry_after_seconds: float | None = None,
) -> RetryResolution:
    """Resolve retry behavior for an experiment cell failure.

    Resolution order:
    1. Provider ``retry-after`` is evaluated first and handled separately from
       normal retry exhaustion.
    2. Normal retry logic uses retryability classification and bounded backoff.
    3. Non-retryable or exhausted retries terminate as ``failed``.

    ``retry-after`` semantics:
    - ``wait <= max_retry_after_window_seconds``: honor provider wait.
    - ``wait > max_retry_after_window_seconds``: terminate as ``rate_limited``.
    """
    effective_config = config or ExperimentRetryConfig()

    if provider_retry_after_seconds is not None:
        if not math.isfinite(provider_retry_after_seconds):
            raise ValueError("provider_retry_after_seconds must be finite when provided")

        wait_seconds = max(0.0, provider_retry_after_seconds)
        if wait_seconds <= effective_config.max_retry_after_window_seconds:
            return RetryResolution(
                action=RetryAction.RETRY_AFTER_HINT,
                category=ErrorCategory.RATE_LIMIT,
                wait_seconds=wait_seconds,
                terminal_status=None,
            )

        return RetryResolution(
            action=RetryAction.RATE_LIMITED,
            category=ErrorCategory.RATE_LIMIT,
            wait_seconds=0.0,
            terminal_status=ExperimentCellStatus.RATE_LIMITED,
        )

    effective_policy = effective_config.to_retry_policy(retry_policy)
    category = classify_error(error, provider=provider)
    decision = effective_policy.should_retry(attempt, error)
    if decision is RetryDecision.RETRY:
        return RetryResolution(
            action=RetryAction.RETRY_WITH_BACKOFF,
            category=category,
            wait_seconds=calculate_backoff(effective_policy, attempt),
            terminal_status=None,
        )

    return RetryResolution(
        action=RetryAction.FAIL,
        category=category,
        wait_seconds=0.0,
        terminal_status=ExperimentCellStatus.FAILED,
    )


class SchedulingPolicy(str, Enum):
    """Scheduling strategy for matrix execution."""

    NON_BLOCKING = "non_blocking"
    GROUPED = "grouped"


@dataclass(frozen=True, slots=True)
class ExperimentSchedulingConfig:
    """Scheduling controls for model aliases.

    Attributes:
        policy: ``NON_BLOCKING`` by default so each alias runs independently.
        alias_group_by_name: Optional alias -> group mapping used only when
            ``policy`` is ``GROUPED``.

    Grouped barriers are scoped to the group only. Aliases in other groups, or
    aliases with no group assignment, continue independently.
    """

    policy: SchedulingPolicy = SchedulingPolicy.NON_BLOCKING
    alias_group_by_name: Mapping[str, str] = field(default_factory=dict)

    def barrier_scope(self, alias: str, all_aliases: Collection[str]) -> frozenset[str]:
        """Return aliases that share execution progress with ``alias``.

        - ``NON_BLOCKING``: only ``alias`` is in scope.
        - ``GROUPED``: aliases in the same configured group share the barrier.

        This keeps one slow provider/model isolated from unrelated aliases unless
        an explicit group assignment ties them together.
        """
        if self.policy is SchedulingPolicy.NON_BLOCKING:
            return frozenset({alias})

        group = self.alias_group_by_name.get(alias)
        if group is None:
            return frozenset({alias})

        scope = {
            candidate
            for candidate in all_aliases
            if self.alias_group_by_name.get(candidate) == group
        }
        if not scope:
            return frozenset({alias})
        return frozenset(scope)


def is_await_all_complete(
    status_matrix: Mapping[str, Mapping[str, ExperimentCellStatus]],
    *,
    prompt_ids: Collection[str],
    aliases: Collection[str],
) -> bool:
    """Return ``True`` only when the full N x M matrix is terminal.

    ``await all`` is defined strictly as full matrix completion:
    every ``(prompt_id, alias)`` cell must exist and be terminal. Completion is
    never inferred from a subset of columns/providers.
    """
    for prompt_id in prompt_ids:
        row = status_matrix.get(prompt_id)
        if row is None:
            return False
        for alias in aliases:
            cell_status = row.get(alias)
            if cell_status is None or not cell_status.is_terminal:
                return False

    return True
