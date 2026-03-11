"""Structured JSONL logging for inference events with secret redaction."""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Patterns for common secrets
SECRET_PATTERNS = [
    # OpenAI API keys (sk-...) - match 10+ chars after sk-
    (r"sk-[a-zA-Z0-9]{10,}", "[REDACTED]"),
    # OpenAI project keys (sk-proj-...)
    (r"sk-proj-[a-zA-Z0-9]{20,}", "[REDACTED]"),
    # Anthropic API keys
    (r"sk-ant-[a-zA-Z0-9-]{20,}", "[REDACTED]"),
    # Bearer tokens - match "Bearer <token>" or "Bearer token <token>"
    (r"Bearer\s+(?:token\s+)?[a-zA-Z0-9_-]{6,}", "Bearer [REDACTED]"),
    # Generic long hex strings (32+ chars, likely API keys)
    (r"\b[a-f0-9]{32,}\b", "[REDACTED]"),
    # Generic alphanumeric tokens (40+ chars)
    (r"\b[a-zA-Z0-9]{40,}\b", "[REDACTED]"),
]


def redact_secrets(text: str) -> str:
    """Redact common secret patterns from text.

    Args:
        text: Text that may contain secrets

    Returns:
        Text with secrets replaced by [REDACTED]
    """
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


@dataclass
class LogEntry:
    """Structured log entry for inference events.

    Required fields:
        provider: LLM provider name (e.g., "openai", "anthropic")
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        status: "success" or "failure"
        latency_ms: Request latency in milliseconds

    Optional fields:
        timestamp: ISO 8601 timestamp (auto-generated if not provided)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used
        cost_usd: Cost in USD
        error_type: Exception/error type for failures
        error_message: Error message (will be redacted)
        retry_count: Number of retries attempted
    """

    provider: str
    model: str
    status: str
    latency_ms: float
    timestamp: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int | None = None

    def __post_init__(self) -> None:
        """Auto-generate timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Redact secrets in error message
        if self.error_message:
            self.error_message = redact_secrets(self.error_message)

    def to_json(self) -> str:
        """Serialize entry to JSON string, omitting None fields.

        Returns:
            JSON string representation
        """
        data = asdict(self)

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        # Remove retry_count if it's 0
        if data.get("retry_count") == 0:
            data.pop("retry_count", None)

        return json.dumps(data, separators=(",", ":"))


class InferenceLogger:
    """Append-safe JSONL logger for inference events.

    Writes one JSON object per line to a file, appending to existing content.
    Thread-safe for concurrent writes to the same file.
    """

    def __init__(self, log_file: Path) -> None:
        """Initialize logger with target file.

        Args:
            log_file: Path to JSONL log file
        """
        self.log_file = Path(log_file)
        # Ensure parent directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: LogEntry) -> None:
        """Write a log entry to the file.

        Args:
            entry: LogEntry to write
        """
        json_line = entry.to_json()

        # Append to file with newline
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")


def log_success(
    log_file: Path,
    provider: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    retry_count: int | None = None,
) -> None:
    """Convenience function to log a successful inference.

    Args:
        log_file: Path to JSONL log file
        provider: LLM provider name
        model: Model identifier
        latency_ms: Request latency in milliseconds
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        cost_usd: Cost in USD
        retry_count: Number of retries attempted
    """
    entry = LogEntry(
        provider=provider,
        model=model,
        status="success",
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        retry_count=retry_count,
    )

    logger = InferenceLogger(log_file)
    logger.write(entry)


def log_failure(
    log_file: Path,
    provider: str,
    model: str,
    latency_ms: float,
    error_type: str,
    error_message: str,
    retry_count: int | None = None,
) -> None:
    """Convenience function to log a failed inference.

    Args:
        log_file: Path to JSONL log file
        provider: LLM provider name
        model: Model identifier
        latency_ms: Request latency in milliseconds
        error_type: Exception/error type
        error_message: Error message (will be redacted)
        retry_count: Number of retries attempted
    """
    entry = LogEntry(
        provider=provider,
        model=model,
        status="failure",
        latency_ms=latency_ms,
        error_type=error_type,
        error_message=error_message,
        retry_count=retry_count,
    )

    logger = InferenceLogger(log_file)
    logger.write(entry)
