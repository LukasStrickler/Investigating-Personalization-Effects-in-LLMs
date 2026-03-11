"""Retry policy and error taxonomy for inference operations.

This module provides:
- Normalized error taxonomy distinguishing fatal vs retryable failures
- Bounded exponential backoff with jitter for retryable failures
- Retry metadata for logging and client observability
"""

import random
import re
from dataclasses import dataclass, field
from enum import Enum, auto


class ErrorCategory(Enum):
    """Classification of error types for retry decisions."""

    AUTH_FAILURE = auto()
    INVALID_REQUEST = auto()
    RATE_LIMIT = auto()
    TIMEOUT = auto()
    NETWORK_ERROR = auto()
    SERVER_ERROR = auto()
    UNKNOWN = auto()

    @property
    def is_retryable(self) -> bool:
        """Return True if errors of this category should be retried."""
        return self in {
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.SERVER_ERROR,
        }


class RetryDecision(Enum):
    """Decision on whether to retry after an error."""

    RETRY = auto()
    STOP = auto()


@dataclass
class RetryPolicy:
    """Configuration for retry behavior with exponential backoff."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    seed: int | None = None

    _rng: random.Random | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize random number generator for deterministic jitter."""
        if self.jitter and self.seed is not None:
            object.__setattr__(self, "_rng", random.Random(self.seed))

    def should_retry(self, attempt: int, error: Exception) -> RetryDecision:
        """
        Determine whether to retry based on attempt count and error type.

        Args:
            attempt: Current attempt number (1-indexed)
            error: The exception that occurred

        Returns:
            RetryDecision.RETRY if should retry, RetryDecision.STOP otherwise
        """
        # Check if error is retryable
        category = classify_error(error)
        if not category.is_retryable:
            return RetryDecision.STOP

        # Check if we've exceeded max retries
        if attempt >= self.max_retries:
            return RetryDecision.STOP

        return RetryDecision.RETRY

    def clone_with_seed(self, seed: int) -> "RetryPolicy":
        """Create a copy of this policy with a new random seed."""
        return RetryPolicy(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
            seed=seed,
        )


@dataclass
class RetryMetadata:
    """Metadata about a retry attempt for observability and logging."""

    attempt: int
    error: Exception
    category: ErrorCategory
    backoff_seconds: float

    def to_dict(self) -> dict:
        """
        Convert metadata to a dictionary for logging.

        Sanitizes error messages to remove potentially sensitive information.
        """
        error_message = str(self.error)

        # Sanitize potential API keys, tokens, passwords
        sanitized_message = self._sanitize_message(error_message)

        return {
            "attempt": self.attempt,
            "category": self.category.name.lower(),
            "backoff_seconds": self.backoff_seconds,
            "error_type": type(self.error).__name__,
            "error_message": sanitized_message,
        }

    @staticmethod
    def _sanitize_message(message: str) -> str:
        """Remove potentially sensitive information from error messages."""
        # Pattern for common API key formats
        patterns = [
            # sk-xxx (OpenAI style) - catch keys with 5+ chars
            (r"sk-[a-zA-Z0-9]{5,}", "sk-***REDACTED***"),
            # Generic long alphanumeric strings that look like keys
            (
                r'(api[_-]?key|token|password|secret)["\s:=]+["\']?[a-zA-Z0-9_-]{5,}["\']?',
                r"\1: ***REDACTED***",
            ),
        ]

        sanitized = message
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized


def classify_error(error: Exception, provider: str | None = None) -> ErrorCategory:
    """
    Classify an error into a normalized category.

    Args:
        error: The exception to classify
        provider: Optional provider name for provider-specific parsing

    Returns:
        The appropriate ErrorCategory
    """
    error_message = str(error).lower()


    # Provider-specific patterns
    if provider:
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return _classify_openai_error(error, error_message)
        elif provider_lower == "anthropic":
            return _classify_anthropic_error(error, error_message)

    # Generic classification based on error type and message

    # Auth failures
    if any(
        x in error_message for x in ["invalid api key", "unauthorized", "authentication", "401"]
    ):
        return ErrorCategory.AUTH_FAILURE

    if any(x in error_message for x in ["forbidden", "access denied", "permission", "403"]):
        return ErrorCategory.AUTH_FAILURE

    # Invalid requests
    if any(x in error_message for x in ["invalid request", "bad request", "malformed", "400"]):
        return ErrorCategory.INVALID_REQUEST

    if any(x in error_message for x in ["missing required", "invalid parameter", "must be"]):
        return ErrorCategory.INVALID_REQUEST

    # Rate limits
    if any(x in error_message for x in ["rate limit", "too many requests", "429"]):
        return ErrorCategory.RATE_LIMIT

    # Timeouts
    if isinstance(error, TimeoutError) or "timeout" in error_message:
        return ErrorCategory.TIMEOUT

    # Network errors
    if isinstance(error, (ConnectionError, ConnectionRefusedError, ConnectionResetError)):
        return ErrorCategory.NETWORK_ERROR

    if any(x in error_message for x in ["network", "connection", "unreachable", "dns"]):
        return ErrorCategory.NETWORK_ERROR

    # Server errors
    if any(
        x in error_message
        for x in ["500", "502", "503", "504", "internal server error", "service unavailable"]
    ):
        return ErrorCategory.SERVER_ERROR

    if any(x in error_message for x in ["overloaded", "capacity", "temporarily unavailable"]):
        return ErrorCategory.SERVER_ERROR

    # Unknown - default to fatal for safety
    return ErrorCategory.UNKNOWN


def _classify_openai_error(error: Exception, error_message: str) -> ErrorCategory:
    """Classify OpenAI-specific errors."""
    # Check for status codes
    if "429" in error_message or "rate limit" in error_message:
        return ErrorCategory.RATE_LIMIT

    if "401" in error_message or "unauthorized" in error_message:
        return ErrorCategory.AUTH_FAILURE

    if "400" in error_message or "invalid" in error_message:
        return ErrorCategory.INVALID_REQUEST

    if "500" in error_message or "502" in error_message or "503" in error_message:
        return ErrorCategory.SERVER_ERROR

    # Fall back to generic classification
    return classify_error(error, provider=None)


def _classify_anthropic_error(error: Exception, error_message: str) -> ErrorCategory:
    """Classify Anthropic-specific errors."""
    # Anthropic-specific patterns
    if "overloaded" in error_message:
        return ErrorCategory.SERVER_ERROR

    if "rate limit" in error_message:
        return ErrorCategory.RATE_LIMIT

    if "invalid" in error_message and ("request" in error_message or "parameter" in error_message):
        return ErrorCategory.INVALID_REQUEST

    if "401" in error_message or "authentication" in error_message:
        return ErrorCategory.AUTH_FAILURE

    # Fall back to generic classification
    return classify_error(error, provider=None)


def calculate_backoff(policy: RetryPolicy, attempt: int) -> float:
    """
    Calculate the backoff delay for a given attempt.

    Uses exponential backoff with optional jitter.

    Args:
        policy: The retry policy configuration
        attempt: The current attempt number (1-indexed)

    Returns:
        The backoff delay in seconds
    """
    # Calculate base exponential backoff
    exponential_delay = policy.base_delay * (policy.exponential_base ** (attempt - 1))

    # Cap at max_delay
    delay = min(exponential_delay, policy.max_delay)

    # Add jitter if enabled
    if policy.jitter:
        if policy._rng is not None:
            # Use seeded RNG for deterministic testing
            jitter_factor = policy._rng.uniform(0.5, 1.5)
        else:
            # Use system random for production
            jitter_factor = random.uniform(0.5, 1.5)

        delay = delay * jitter_factor

        # Ensure we don't exceed max_delay after jitter
        delay = min(delay, policy.max_delay)

    return delay
