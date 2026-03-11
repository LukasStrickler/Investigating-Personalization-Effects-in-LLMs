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
    system_prompt: str | None = None
    messages: list[dict] | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
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
    tool_calls: list[dict] | None = None
    """Optional metadata from the provider (e.g. system_prompt_folded + system_prompt when applied)."""
    metadata: dict | None = None


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
        self._concurrency_limiter = _ProviderConcurrencyLimiter(self._providers_by_name)
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
        await self._concurrency_limiter.acquire(alias_cfg.provider, request.model_alias)
        try:
            return await self._complete_impl(request, alias_cfg, provider_cfg)
        finally:
            self._concurrency_limiter.release(alias_cfg.provider, request.model_alias)

    async def _complete_impl(
        self,
        request: InferenceRequest,
        alias_cfg: ModelAliasConfig,
        provider_cfg: ProviderConfig,
    ) -> InferenceResult:
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
                    system_prompt=request.system_prompt,
                    messages=request.messages,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
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
                tool_calls=provider_response.tool_calls,
                metadata=provider_response.metadata,
            )

    def get_provider_model(self, model_alias: str) -> tuple[str, str] | None:
        """Return (provider, model) for an alias, or None if alias is not configured."""
        alias_cfg = self._config.model_aliases.get(model_alias)
        if alias_cfg is None:
            return None
        return (alias_cfg.provider, alias_cfg.model)

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


class _ProviderConcurrencyLimiter:
    """Limits concurrent requests per provider (and optionally per model) from config."""

    def __init__(self, providers_by_name: dict[str, ProviderConfig]) -> None:
        self._providers = providers_by_name
        self._semaphores: dict[tuple[str, str | None], asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    def _key(self, provider: str, model_alias: str | None) -> tuple[str, str | None]:
        cfg = self._providers.get(provider)
        if cfg and cfg.per_model_concurrency > 0 and model_alias is not None:
            return (provider, model_alias)
        return (provider, None)

    async def acquire(self, provider: str, model_alias: str | None = None) -> None:
        cfg = self._providers.get(provider)
        if not cfg or (cfg.max_concurrency <= 0 and cfg.per_model_concurrency <= 0):
            return
        key = self._key(provider, model_alias)
        limit = cfg.per_model_concurrency if key[1] is not None else cfg.max_concurrency
        if limit <= 0:
            return
        async with self._lock:
            if key not in self._semaphores:
                self._semaphores[key] = asyncio.Semaphore(limit)
        await self._semaphores[key].acquire()

    def release(self, provider: str, model_alias: str | None = None) -> None:
        key = self._key(provider, model_alias)
        if key in self._semaphores:
            self._semaphores[key].release()


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
