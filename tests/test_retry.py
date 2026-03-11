"""Tests for retry policy and error taxonomy."""

from inference.retry import (
    ErrorCategory,
    RetryDecision,
    RetryMetadata,
    RetryPolicy,
    calculate_backoff,
    classify_error,
)


class TestErrorTaxonomy:
    """Test error classification into fatal vs retryable categories."""

    def test_auth_failure_is_fatal(self):
        """Authentication failures should not be retried."""
        error = Exception("Invalid API key")
        category = classify_error(error)
        assert category == ErrorCategory.AUTH_FAILURE
        assert not category.is_retryable

    def test_invalid_request_is_fatal(self):
        """Malformed requests should not be retried."""
        error = Exception("Invalid request: missing required parameter")
        category = classify_error(error)
        assert category == ErrorCategory.INVALID_REQUEST
        assert not category.is_retryable

    def test_rate_limit_is_retryable(self):
        """Rate limit errors should be retried with backoff."""
        error = Exception("Rate limit exceeded")
        category = classify_error(error)
        assert category == ErrorCategory.RATE_LIMIT
        assert category.is_retryable

    def test_timeout_is_retryable(self):
        """Timeout errors should be retried."""
        error = TimeoutError("Request timed out")
        category = classify_error(error)
        assert category == ErrorCategory.TIMEOUT
        assert category.is_retryable

    def test_network_error_is_retryable(self):
        """Network connectivity errors should be retried."""
        error = ConnectionError("Network unreachable")
        category = classify_error(error)
        assert category == ErrorCategory.NETWORK_ERROR
        assert category.is_retryable

    def test_server_error_is_retryable(self):
        """5xx server errors should be retried."""
        error = Exception("Internal server error (500)")
        category = classify_error(error)
        assert category == ErrorCategory.SERVER_ERROR
        assert category.is_retryable

    def test_unknown_error_is_fatal_by_default(self):
        """Unknown errors default to fatal for safety."""
        error = RuntimeError("Unexpected state")
        category = classify_error(error)
        assert category == ErrorCategory.UNKNOWN
        assert not category.is_retryable


class TestRetryPolicy:
    """Test retry policy with bounded exponential backoff."""

    def test_retry_policy_default_values(self):
        """Retry policy should have sensible defaults."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0

    def test_should_retry_within_limit(self):
        """Should allow retry when under max retries."""
        policy = RetryPolicy(max_retries=3)
        decision = policy.should_retry(attempt=1, error=Exception("Rate limit"))
        assert decision == RetryDecision.RETRY

    def test_should_not_retry_beyond_limit(self):
        """Should not retry when max retries exceeded."""
        policy = RetryPolicy(max_retries=3)
        decision = policy.should_retry(attempt=3, error=Exception("Rate limit"))
        assert decision == RetryDecision.STOP

    def test_should_not_retry_fatal_error(self):
        """Should not retry fatal errors regardless of attempt count."""
        policy = RetryPolicy(max_retries=3)
        decision = policy.should_retry(attempt=1, error=Exception("Invalid API key"))
        assert decision == RetryDecision.STOP

    def test_backoff_increases_exponentially(self):
        """Backoff should increase exponentially with attempt."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False)
        assert calculate_backoff(policy, attempt=1) == 1.0
        assert calculate_backoff(policy, attempt=2) == 2.0
        assert calculate_backoff(policy, attempt=3) == 4.0
        assert calculate_backoff(policy, attempt=4) == 8.0

    def test_backoff_respects_max_delay(self):
        """Backoff should cap at max_delay."""
        policy = RetryPolicy(base_delay=10.0, exponential_base=2.0, max_delay=30.0, jitter=False)
        # Without cap: 10, 20, 40, 80...
        assert calculate_backoff(policy, attempt=1) == 10.0
        assert calculate_backoff(policy, attempt=2) == 20.0
        assert calculate_backoff(policy, attempt=3) == 30.0  # capped
        assert calculate_backoff(policy, attempt=4) == 30.0  # capped

    def test_backoff_with_jitter_is_deterministic_with_seed(self):
        """Backoff with jitter should be deterministic when seeded."""
        policy = RetryPolicy(
            base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=True, seed=42
        )
        # Same seed should produce same jitter sequence
        backoff1_a = calculate_backoff(policy, attempt=1)
        backoff1_b = calculate_backoff(policy.clone_with_seed(42), attempt=1)
        assert backoff1_a == backoff1_b

    def test_backoff_jitter_stays_in_bounds(self):
        """Jitter should keep backoff within reasonable bounds."""
        policy = RetryPolicy(
            base_delay=1.0, exponential_base=2.0, max_delay=600.0, jitter=True, seed=123
        )
        for attempt in range(1, 10):
            backoff = calculate_backoff(policy, attempt)
            base_backoff = policy.base_delay * (policy.exponential_base ** (attempt - 1))
            # Jitter should typically be within ±50% of base backoff
            # Note: max_delay cap may reduce final value if base_backoff > max_delay
            expected_max = min(base_backoff * 1.5, policy.max_delay)
            expected_min = min(base_backoff * 0.5, policy.max_delay * 0.5)
            assert expected_min <= backoff <= expected_max


class TestRetryMetadata:
    """Test retry metadata for observability."""

    def test_metadata_captures_attempt_info(self):
        """Metadata should capture attempt number and error."""
        error = Exception("Rate limit")
        metadata = RetryMetadata(
            attempt=2,
            error=error,
            category=ErrorCategory.RATE_LIMIT,
            backoff_seconds=2.0,
        )
        assert metadata.attempt == 2
        assert metadata.error == error
        assert metadata.category == ErrorCategory.RATE_LIMIT
        assert metadata.backoff_seconds == 2.0

    def test_metadata_is_serializable(self):
        """Metadata should be convertible to dict for logging."""
        error = Exception("Timeout")
        metadata = RetryMetadata(
            attempt=1,
            error=error,
            category=ErrorCategory.TIMEOUT,
            backoff_seconds=1.5,
        )
        result = metadata.to_dict()
        assert isinstance(result, dict)
        assert result["attempt"] == 1
        assert result["category"] == "timeout"
        assert result["backoff_seconds"] == 1.5
        assert "error_type" in result
        assert "error_message" in result

    def test_metadata_hides_sensitive_info(self):
        """Metadata should not expose sensitive error details."""
        error = Exception("Invalid API key: sk-12345")
        metadata = RetryMetadata(
            attempt=1,
            error=error,
            category=ErrorCategory.AUTH_FAILURE,
            backoff_seconds=0.0,
        )
        result = metadata.to_dict()
        # Should not contain the actual key
        assert "sk-12345" not in str(result)


class TestProviderAwareClassification:
    """Test provider-specific error parsing."""

    def test_openai_rate_limit_classification(self):
        """OpenAI rate limit errors should be classified correctly."""
        error = Exception("Error code: 429 - Rate limit exceeded")
        category = classify_error(error, provider="openai")
        assert category == ErrorCategory.RATE_LIMIT

    def test_anthropic_overloaded_classification(self):
        """Anthropic overloaded errors should be classified as retryable."""
        error = Exception("Overloaded")
        category = classify_error(error, provider="anthropic")
        assert category == ErrorCategory.SERVER_ERROR
        assert category.is_retryable

    def test_openai_auth_error_classification(self):
        """OpenAI authentication errors should be fatal."""
        error = Exception("Error code: 401 - Unauthorized")
        category = classify_error(error, provider="openai")
        assert category == ErrorCategory.AUTH_FAILURE
        assert not category.is_retryable

    def test_anthropic_invalid_request_classification(self):
        """Anthropic invalid request errors should be fatal."""
        error = Exception("Invalid request: max_tokens must be positive")
        category = classify_error(error, provider="anthropic")
        assert category == ErrorCategory.INVALID_REQUEST
        assert not category.is_retryable


class TestIntegration:
    """Integration tests for retry flow."""

    def test_full_retry_cycle(self):
        """Test a complete retry cycle with metadata tracking."""
        policy = RetryPolicy(max_retries=3, jitter=False, seed=42)
        error = Exception("Rate limit exceeded")

        # First attempt - should retry
        decision1 = policy.should_retry(attempt=1, error=error)
        assert decision1 == RetryDecision.RETRY

        # Calculate backoff for first retry
        backoff1 = calculate_backoff(policy, attempt=1)
        metadata1 = RetryMetadata(
            attempt=1,
            error=error,
            category=classify_error(error),
            backoff_seconds=backoff1,
        )
        assert metadata1.category.is_retryable

        # After max retries - should stop
        decision_final = policy.should_retry(attempt=3, error=error)
        assert decision_final == RetryDecision.STOP

    def test_fatal_error_stops_immediately(self):
        """Fatal errors should stop retry immediately."""
        policy = RetryPolicy(max_retries=3)
        error = Exception("Invalid API key")

        decision = policy.should_retry(attempt=1, error=error)
        assert decision == RetryDecision.STOP

        category = classify_error(error)
        assert not category.is_retryable
