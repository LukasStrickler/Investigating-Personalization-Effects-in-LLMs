from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from inference.config import MOCK_PROVIDER, load_config_from_file, resolve_api_key
from inference.logging import log_failure, log_success
from inference.providers import LiteLLMProviderAdapter, ProviderAdapter, ProviderRequest
from inference.rate_limits import ProviderRateLimiter, RateLimitPolicy
from inference.retry import (
    RetryDecision,
    RetryMetadata,
    RetryPolicy,
    calculate_backoff,
    classify_error,
)
from inference.types import (
    InferenceConfig,
    ModelAliasConfig,
    ProviderConfig,
    RateLimit,
    RetryConfig,
)

SleepFn = Callable[[float], Awaitable[None]]


class RateLimiterProtocol(Protocol):
    def configure(self, provider: str, profile: str, policy: RateLimitPolicy) -> None: ...

    async def acquire(
        self,
        provider: str,
        *,
        profile: str = "default",
        tokens: int = 0,
        wait: bool = True,
    ) -> float: ...


class InferenceClientError(RuntimeError):
    pass


class UnknownModelAliasError(InferenceClientError):
    pass


class InferenceRequestError(InferenceClientError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    model_alias: str
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    model_alias: str
    provider: str
    model: str
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    retry_count: int


class UnifiedInferenceClient:
    def __init__(
        self,
        *,
        config: InferenceConfig,
        adapter: ProviderAdapter | None = None,
        limiter: RateLimiterProtocol | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self._config = config
        self._adapter = adapter or LiteLLMProviderAdapter()
        self._limiter = limiter or ProviderRateLimiter()
        self._sleep = sleep or asyncio.sleep
        self._log_path = Path(config.log_path) if config.log_path else None
        self._providers_by_name = _index_providers(config)
        self._configure_rate_limits()

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        *,
        adapter: ProviderAdapter | None = None,
        limiter: RateLimiterProtocol | None = None,
        sleep: SleepFn | None = None,
    ) -> UnifiedInferenceClient:
        config = load_config_from_file(config_path)
        return cls(config=config, adapter=adapter, limiter=limiter, sleep=sleep)

    async def complete(self, request: InferenceRequest) -> InferenceResult:
        alias_cfg = self._resolve_alias(request.model_alias)
        provider_cfg = self._resolve_provider(alias_cfg.provider)
        retry_policy = _to_retry_policy(provider_cfg.retry or self._config.default_retry)

        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            estimated_tokens = _estimate_tokens(request)
            await self._limiter.acquire(
                alias_cfg.provider,
                profile="default",
                tokens=estimated_tokens,
                wait=True,
            )

            try:
                provider_request = ProviderRequest(
                    provider=alias_cfg.provider,
                    model=alias_cfg.model,
                    prompt=request.prompt,
                    api_key=self._resolve_api_key(provider_cfg),
                    base_url=provider_cfg.base_url,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                provider_response = await self._adapter.complete(provider_request)
            except Exception as error:
                category = classify_error(error, provider=alias_cfg.provider)
                decision = retry_policy.should_retry(attempt, error)
                if decision is RetryDecision.RETRY:
                    backoff_seconds = calculate_backoff(retry_policy, attempt)
                    _ = RetryMetadata(
                        attempt=attempt,
                        error=error,
                        category=category,
                        backoff_seconds=backoff_seconds,
                    )
                    await self._sleep(backoff_seconds)
                    continue

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                metadata = RetryMetadata(
                    attempt=attempt,
                    error=error,
                    category=category,
                    backoff_seconds=0.0,
                )
                if self._log_path is not None:
                    log_failure(
                        log_file=self._log_path,
                        provider=alias_cfg.provider,
                        model=alias_cfg.model,
                        latency_ms=elapsed_ms,
                        error_type=type(error).__name__,
                        error_message=metadata.to_dict()["error_message"],
                        retry_count=attempt - 1,
                    )
                raise InferenceRequestError(
                    f"Inference failed for alias={request.model_alias!r}, "
                    f"provider={alias_cfg.provider!r}, model={alias_cfg.model!r}"
                ) from error

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if self._log_path is not None:
                log_success(
                    log_file=self._log_path,
                    provider=alias_cfg.provider,
                    model=alias_cfg.model,
                    latency_ms=elapsed_ms,
                    prompt_tokens=provider_response.prompt_tokens,
                    completion_tokens=provider_response.completion_tokens,
                    total_tokens=provider_response.total_tokens,
                    retry_count=attempt - 1,
                )

            return InferenceResult(
                model_alias=request.model_alias,
                provider=alias_cfg.provider,
                model=alias_cfg.model,
                content=provider_response.content,
                prompt_tokens=provider_response.prompt_tokens,
                completion_tokens=provider_response.completion_tokens,
                total_tokens=provider_response.total_tokens,
                latency_ms=elapsed_ms,
                retry_count=attempt - 1,
            )

    def _resolve_alias(self, model_alias: str) -> ModelAliasConfig:
        alias = self._config.model_aliases.get(model_alias)
        if alias is None:
            raise UnknownModelAliasError(
                f"Unknown model alias {model_alias!r}. "
                f"Configured aliases: {sorted(self._config.model_aliases)}"
            )
        return alias

    def _resolve_provider(self, provider_name: str) -> ProviderConfig:
        provider = self._providers_by_name.get(provider_name)
        if provider is None:
            raise InferenceClientError(
                f"Model alias resolved to unknown provider {provider_name!r}. "
                f"Configured providers: {sorted(self._providers_by_name)}"
            )
        return provider

    def _resolve_api_key(self, provider: ProviderConfig) -> str | None:
        if provider.name == MOCK_PROVIDER:
            return None
        return resolve_api_key(provider)

    def _configure_rate_limits(self) -> None:
        for provider_name, provider_cfg in self._providers_by_name.items():
            self._limiter.configure(
                provider_name,
                "default",
                _to_rate_limit_policy(provider_cfg.rate_limit),
            )


def _index_providers(config: InferenceConfig) -> dict[str, ProviderConfig]:
    indexed: dict[str, ProviderConfig] = {}
    for provider in config.providers.values():
        indexed[provider.name] = provider
    return indexed


def _to_rate_limit_policy(rate_limit: RateLimit | None) -> RateLimitPolicy:
    if rate_limit is None:
        return RateLimitPolicy()
    return RateLimitPolicy(
        requests_per_minute=rate_limit.requests_per_minute or None,
        tokens_per_minute=rate_limit.tokens_per_minute or None,
    )


def _to_retry_policy(retry: RetryConfig | None) -> RetryPolicy:
    if retry is None:
        return RetryPolicy()
    return RetryPolicy(
        max_retries=retry.max_retries,
        base_delay=retry.base_delay,
        max_delay=retry.max_delay,
    )


def _estimate_tokens(request: InferenceRequest) -> int:
    prompt_estimate = max(0, len(request.prompt.split()))
    completion_estimate = max(0, request.max_tokens or 0)
    return prompt_estimate + completion_estimate


__all__ = [
    "InferenceClientError",
    "InferenceRequest",
    "InferenceRequestError",
    "InferenceResult",
    "UnifiedInferenceClient",
    "UnknownModelAliasError",
]
