"""Tests for structured JSONL logging foundation."""

import json
from datetime import datetime, timezone

from inference.logging import (
    InferenceLogger,
    LogEntry,
    log_failure,
    log_success,
)


class TestLogEntrySchema:
    """Test the log entry schema structure."""

    def test_success_entry_has_required_fields(self):
        """Success entries must have provider, model, status, timestamp, latency, tokens."""
        entry = LogEntry(
            provider="openai",
            model="gpt-4",
            status="success",
            latency_ms=1250.5,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.003,
        )

        assert entry.provider == "openai"
        assert entry.model == "gpt-4"
        assert entry.status == "success"
        assert entry.latency_ms == 1250.5
        assert entry.prompt_tokens == 100
        assert entry.completion_tokens == 50
        assert entry.total_tokens == 150
        assert entry.cost_usd == 0.003
        assert entry.timestamp is not None

    def test_failure_entry_has_error_fields(self):
        """Failure entries must have error_type and error_message."""
        entry = LogEntry(
            provider="anthropic",
            model="claude-3-opus",
            status="failure",
            latency_ms=500.0,
            error_type="RateLimitError",
            error_message="Rate limit exceeded",
        )

        assert entry.status == "failure"
        assert entry.error_type == "RateLimitError"
        assert entry.error_message == "Rate limit exceeded"
        assert entry.prompt_tokens is None
        assert entry.completion_tokens is None

    def test_entry_serializes_to_json(self):
        """Log entries must serialize to valid JSON."""
        entry = LogEntry(
            provider="openai",
            model="gpt-3.5-turbo",
            status="success",
            latency_ms=800.0,
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            cost_usd=0.001,
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["provider"] == "openai"
        assert data["model"] == "gpt-3.5-turbo"
        assert data["status"] == "success"
        assert "timestamp" in data

    def test_entry_omits_none_fields_in_json(self):
        """Optional None fields should be omitted from JSON output."""
        entry = LogEntry(
            provider="mock",
            model="mock-model",
            status="success",
            latency_ms=100.0,
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert "prompt_tokens" not in data
        assert "completion_tokens" not in data
        assert "error_type" not in data
        assert "error_message" not in data


class TestSecretRedaction:
    """Test that secrets are redacted from logs."""

    def test_redacts_api_key_in_error_message(self):
        """API keys must be redacted from error messages."""
        entry = LogEntry(
            provider="openai",
            model="gpt-4",
            status="failure",
            latency_ms=100.0,
            error_type="AuthenticationError",
            error_message="Invalid API key: sk-1234567890abcdef",
        )

        json_str = entry.to_json()

        assert "sk-1234567890abcdef" not in json_str
        assert "[REDACTED]" in json_str

    def test_redacts_bearer_token(self):
        """Bearer tokens must be redacted."""
        entry = LogEntry(
            provider="anthropic",
            model="claude-3-sonnet",
            status="failure",
            latency_ms=200.0,
            error_type="AuthError",
            error_message="Bearer token abc123xyz expired",
        )

        json_str = entry.to_json()

        assert "abc123xyz" not in json_str
        assert "[REDACTED]" in json_str

    def test_redacts_long_secrets(self):
        """Long API keys (40+ chars) must be redacted."""
        long_key = "sk-proj-" + "a" * 48
        entry = LogEntry(
            provider="openai",
            model="gpt-4-turbo",
            status="failure",
            latency_ms=150.0,
            error_type="KeyError",
            error_message=f"Key {long_key} is invalid",
        )

        json_str = entry.to_json()

        assert long_key not in json_str


class TestJSONLWriting:
    """Test JSONL file writing behavior."""

    def test_writes_single_line_json(self, tmp_path):
        """Each log entry must be written as a single JSON line."""
        log_file = tmp_path / "inference.log"
        logger = InferenceLogger(log_file)

        entry = LogEntry(
            provider="openai",
            model="gpt-4",
            status="success",
            latency_ms=1000.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.002,
        )

        logger.write(entry)

        content = log_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 1
        # Verify it's valid JSON
        data = json.loads(lines[0])
        assert data["provider"] == "openai"

    def test_appends_to_existing_file(self, tmp_path):
        """Logger must append to existing log file without overwriting."""
        log_file = tmp_path / "inference.log"
        log_file.write_text('{"existing": "entry"}\n')

        logger = InferenceLogger(log_file)
        entry = LogEntry(
            provider="anthropic",
            model="claude-3",
            status="success",
            latency_ms=900.0,
        )
        logger.write(entry)

        content = log_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": "entry"}
        assert json.loads(lines[1])["provider"] == "anthropic"

    def test_handles_concurrent_appends(self, tmp_path):
        """Multiple logger instances can append safely."""
        log_file = tmp_path / "inference.log"

        # Create two loggers pointing to same file
        logger1 = InferenceLogger(log_file)
        logger2 = InferenceLogger(log_file)

        entry1 = LogEntry(provider="openai", model="gpt-4", status="success", latency_ms=100.0)
        entry2 = LogEntry(
            provider="anthropic", model="claude-3", status="success", latency_ms=200.0
        )

        logger1.write(entry1)
        logger2.write(entry2)

        content = log_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        assert json.loads(lines[0])["provider"] == "openai"
        assert json.loads(lines[1])["provider"] == "anthropic"


class TestConvenienceFunctions:
    """Test convenience functions for common logging patterns."""

    def test_log_success_helper(self, tmp_path):
        """log_success must create and write a success entry."""
        log_file = tmp_path / "inference.log"

        log_success(
            log_file=log_file,
            provider="openai",
            model="gpt-4",
            latency_ms=1234.5,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.003,
        )

        content = log_file.read_text()
        data = json.loads(content.strip())

        assert data["status"] == "success"
        assert data["provider"] == "openai"
        assert data["latency_ms"] == 1234.5

    def test_log_failure_helper(self, tmp_path):
        """log_failure must create and write a failure entry."""
        log_file = tmp_path / "inference.log"

        log_failure(
            log_file=log_file,
            provider="anthropic",
            model="claude-3-opus",
            latency_ms=500.0,
            error_type="RateLimitError",
            error_message="Rate limit exceeded - try again in 60s",
        )

        content = log_file.read_text()
        data = json.loads(content.strip())

        assert data["status"] == "failure"
        assert data["error_type"] == "RateLimitError"
        assert "Rate limit exceeded" in data["error_message"]


class TestTimestampFormat:
    """Test timestamp handling."""

    def test_auto_generates_iso_timestamp(self):
        """Entries must auto-generate ISO 8601 timestamp if not provided."""
        before = datetime.now(timezone.utc)
        entry = LogEntry(
            provider="openai",
            model="gpt-4",
            status="success",
            latency_ms=100.0,
        )
        after = datetime.now(timezone.utc)

        # Parse the timestamp
        assert entry.timestamp is not None  # Auto-generated for success entries
        entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))

        # Should be between before and after
        assert (
            before.replace(tzinfo=None)
            <= entry_time.replace(tzinfo=None)
            <= after.replace(tzinfo=None)
        )

    def test_accepts_custom_timestamp(self):
        """Entries must accept custom timestamp."""
        custom_time = "2024-01-15T10:30:00Z"
        entry = LogEntry(
            provider="mock",
            model="mock-model",
            status="success",
            latency_ms=50.0,
            timestamp=custom_time,
        )

        assert entry.timestamp == custom_time

        json_str = entry.to_json()
        data = json.loads(json_str)
        assert data["timestamp"] == custom_time


class TestRetryCount:
    """Test retry count field for resumable runs."""

    def test_includes_retry_count_when_present(self):
        """Entries with retry_count must include it in JSON."""
        entry = LogEntry(
            provider="openai",
            model="gpt-4",
            status="success",
            latency_ms=2000.0,
            retry_count=2,
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["retry_count"] == 2

    def test_omits_retry_count_when_zero(self):
        """Entries with retry_count=0 should omit the field."""
        entry = LogEntry(
            provider="openai",
            model="gpt-4",
            status="success",
            latency_ms=1000.0,
            retry_count=0,
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert "retry_count" not in data
