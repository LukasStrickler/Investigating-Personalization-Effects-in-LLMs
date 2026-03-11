from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference.types import (
    InferenceConfig,
    ModelAliasConfig,
    ProviderConfig,
    RateLimit,
    RetryConfig,
)


class LimiterSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.configured: list[dict[str, Any]] = []

    def configure(self, provider: str, profile: str, policy: Any) -> None:
        self.configured.append(
            {
                "provider": provider,
                "profile": profile,
                "policy": policy,
            }
        )

    async def acquire(
        self,
        provider: str,
        *,
        profile: str = "default",
        tokens: int = 0,
        wait: bool = True,
    ) -> float:
        self.calls.append(
            {
                "provider": provider,
                "profile": profile,
                "tokens": tokens,
                "wait": wait,
            }
        )
        return 0.0


def _build_config(log_path: Path) -> InferenceConfig:
    return InferenceConfig(
        providers={
            "openai": ProviderConfig(
                name="openai",
                api_key_env="OPENAI_API_KEY",
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
            "openrouter": ProviderConfig(
                name="openrouter",
                api_key_env="OPENROUTER_API_KEY",
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
            "mock": ProviderConfig(
                name="mock",
                api_key_env="MOCK_API_KEY",
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
        },
        default_provider="openai",
        model_aliases={
            "research-openai": ModelAliasConfig(
                alias="research-openai",
                provider="openai",
                model="gpt-4o-mini",
            ),
            "research-anthropic": ModelAliasConfig(
                alias="research-anthropic",
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
            ),
            "research-openrouter": ModelAliasConfig(
                alias="research-openrouter",
                provider="openrouter",
                model="openai/gpt-4o-mini",
            ),
            "research-openrouter-prefixed": ModelAliasConfig(
                alias="research-openrouter-prefixed",
                provider="openrouter",
                model="openrouter/meta-llama/llama-3.1-8b-instruct",
            ),
            "research-mock": ModelAliasConfig(
                alias="research-mock",
                provider="mock",
                model="mock-model-v1",
            ),
        },
        default_retry=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.02),
        log_path=str(log_path),
    )


@pytest.mark.asyncio
async def test_same_client_call_works_across_provider_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    models_called: list[str] = []

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        models_called.append(kwargs["model"])
        model_name = kwargs["model"]
        return {
            "choices": [{"message": {"content": f"normalized-{model_name}"}}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
            },
        }

    config = _build_config(tmp_path / "inference.jsonl")
    adapter = LiteLLMProviderAdapter(completion_callable=fake_completion)
    client = UnifiedInferenceClient(config=config, adapter=adapter)

    aliases = [
        "research-openai",
        "research-anthropic",
        "research-openrouter",
        "research-openrouter-prefixed",
    ]
    results = []
    for alias in aliases:
        result = await client.complete(InferenceRequest(model_alias=alias, prompt="hello"))
        results.append(result)

    assert [result.provider for result in results] == [
        "openai",
        "anthropic",
        "openrouter",
        "openrouter",
    ]
    assert all(result.content.startswith("normalized-") for result in results)
    assert all(result.total_tokens == 12 for result in results)
    assert all(not hasattr(result, "raw_response") for result in results)
    assert models_called == [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20241022",
        "openrouter/openai/gpt-4o-mini",
        "openrouter/meta-llama/llama-3.1-8b-instruct",
    ]


@pytest.mark.asyncio
async def test_unknown_provider_alias_fails_before_litellm_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, UnifiedInferenceClient, UnknownModelAliasError
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    call_count = 0

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return kwargs

    config = _build_config(tmp_path / "inference.jsonl")
    adapter = LiteLLMProviderAdapter(completion_callable=fake_completion)
    client = UnifiedInferenceClient(config=config, adapter=adapter)

    with pytest.raises(UnknownModelAliasError) as exc_info:
        _ = await client.complete(InferenceRequest(model_alias="does-not-exist", prompt="hello"))

    assert "does-not-exist" in str(exc_info.value)
    assert call_count == 0


@pytest.mark.asyncio
async def test_client_exercises_limiter_retry_and_logging_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter
    from inference.retry import ErrorCategory

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    classify_calls: list[Exception] = []
    backoff_calls: list[int] = []
    sleep_calls: list[float] = []
    limiter = LimiterSpy()

    def classify_spy(error: Exception, provider: str | None = None) -> ErrorCategory:
        del provider
        classify_calls.append(error)
        return ErrorCategory.RATE_LIMIT

    def backoff_spy(policy: Any, attempt: int) -> float:
        del policy
        backoff_calls.append(attempt)
        return 0.0

    async def sleep_spy(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("inference.client.classify_error", classify_spy)
    monkeypatch.setattr("inference.client.calculate_backoff", backoff_spy)

    call_count = 0

    async def flaky_completion(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("429 rate limit")
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    log_path = tmp_path / "inference.jsonl"
    config = _build_config(log_path)
    client = UnifiedInferenceClient(
        config=config,
        adapter=LiteLLMProviderAdapter(completion_callable=flaky_completion),
        limiter=limiter,
        sleep=sleep_spy,
    )

    result = await client.complete(InferenceRequest(model_alias="research-openai", prompt="hello"))

    assert result.content == "ok"
    assert len(limiter.calls) == 2
    assert len(classify_calls) == 1
    assert backoff_calls == [1]
    assert sleep_calls == [0.0]

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["status"] == "success"
    assert entry["retry_count"] == 1


@pytest.mark.asyncio
async def test_provider_retry_override_wins_over_default_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, InferenceRequestError, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter
    from inference.retry import ErrorCategory

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    config = _build_config(tmp_path / "inference.jsonl")
    openai_cfg = config.providers["openai"]
    providers = {
        **config.providers,
        "openai": ProviderConfig(
            name=openai_cfg.name,
            api_key_env=openai_cfg.api_key_env,
            rate_limit=openai_cfg.rate_limit,
            retry=RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.01),
            base_url=openai_cfg.base_url,
            default_model=openai_cfg.default_model,
        ),
    }
    config = InferenceConfig(
        providers=providers,
        default_provider=config.default_provider,
        model_aliases=config.model_aliases,
        log_path=config.log_path,
        checkpoint_path=config.checkpoint_path,
        default_retry=RetryConfig(max_retries=5, base_delay=0.01, max_delay=0.01),
    )

    attempts = 0

    async def always_fails(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        nonlocal attempts
        attempts += 1
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(
        "inference.client.classify_error", lambda *_args, **_kwargs: ErrorCategory.RATE_LIMIT
    )
    monkeypatch.setattr("inference.client.calculate_backoff", lambda *_args, **_kwargs: 0.0)

    client = UnifiedInferenceClient(
        config=config,
        adapter=LiteLLMProviderAdapter(completion_callable=always_fails),
    )

    with pytest.raises(InferenceRequestError):
        _ = await client.complete(InferenceRequest(model_alias="research-openai", prompt="hello"))

    assert attempts == 1


@pytest.mark.asyncio
async def test_mock_provider_path_does_not_require_api_key_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.delenv("MOCK_API_KEY", raising=False)

    call_count = 0

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return kwargs

    client = UnifiedInferenceClient(
        config=_build_config(tmp_path / "inference.jsonl"),
        adapter=LiteLLMProviderAdapter(completion_callable=fake_completion),
    )

    result = await client.complete(InferenceRequest(model_alias="research-mock", prompt="hello"))

    assert result.provider == "mock"
    assert result.content
    assert call_count == 0
