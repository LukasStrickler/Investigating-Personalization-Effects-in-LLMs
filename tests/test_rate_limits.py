from __future__ import annotations

import asyncio

import pytest

from inference.rate_limits import ProviderRateLimiter, RateLimitExceeded, RateLimitPolicy


class FakeClock:
    def __init__(self) -> None:
        self.current: float = 0.0

    def now(self) -> float:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.current += seconds
        await asyncio.sleep(0)

    def advance(self, seconds: float) -> None:
        self.current += seconds


@pytest.mark.asyncio
async def test_provider_buckets_are_isolated() -> None:
    clock = FakeClock()
    limiter = ProviderRateLimiter(clock=clock.now, sleep=clock.sleep)
    limiter.configure(
        "openai", "gpt-4o-mini", RateLimitPolicy(requests_per_minute=1, tokens_per_minute=100)
    )
    limiter.configure(
        "anthropic", "claude-sonnet", RateLimitPolicy(requests_per_minute=1, tokens_per_minute=100)
    )

    _ = await limiter.acquire("openai", profile="gpt-4o-mini", tokens=10)

    with pytest.raises(RateLimitExceeded):
        _ = await limiter.acquire("openai", profile="gpt-4o-mini", tokens=10, wait=False)

    waited = await limiter.acquire("anthropic", profile="claude-sonnet", tokens=10)
    assert waited == 0.0


@pytest.mark.asyncio
async def test_model_profiles_within_provider_are_isolated() -> None:
    clock = FakeClock()
    limiter = ProviderRateLimiter(clock=clock.now, sleep=clock.sleep)
    limiter.configure(
        "openai", "gpt-4o", RateLimitPolicy(requests_per_minute=1, tokens_per_minute=100)
    )
    limiter.configure(
        "openai", "gpt-4o-mini", RateLimitPolicy(requests_per_minute=1, tokens_per_minute=100)
    )

    _ = await limiter.acquire("openai", profile="gpt-4o", tokens=20)

    with pytest.raises(RateLimitExceeded):
        _ = await limiter.acquire("openai", profile="gpt-4o", tokens=20, wait=False)

    waited = await limiter.acquire("openai", profile="gpt-4o-mini", tokens=20)
    assert waited == 0.0


@pytest.mark.asyncio
async def test_request_and_token_limits_are_both_supported() -> None:
    clock = FakeClock()
    limiter = ProviderRateLimiter(clock=clock.now, sleep=clock.sleep)
    limiter.configure(
        "openrouter", "default", RateLimitPolicy(requests_per_minute=5, tokens_per_minute=10)
    )

    _ = await limiter.acquire("openrouter", tokens=7)

    with pytest.raises(RateLimitExceeded):
        _ = await limiter.acquire("openrouter", tokens=5, wait=False)

    clock.advance(60)
    _ = await limiter.acquire("openrouter", tokens=5)

    for _ in range(4):
        _ = await limiter.acquire("openrouter", tokens=0)

    with pytest.raises(RateLimitExceeded):
        _ = await limiter.acquire("openrouter", tokens=0, wait=False)


@pytest.mark.asyncio
async def test_acquire_waits_with_injected_monotonic_clock() -> None:
    clock = FakeClock()
    limiter = ProviderRateLimiter(clock=clock.now, sleep=clock.sleep)
    limiter.configure(
        "openai", "default", RateLimitPolicy(requests_per_minute=1, tokens_per_minute=100)
    )

    _ = await limiter.acquire("openai", tokens=10)
    waited = await limiter.acquire("openai", tokens=10)

    assert waited == 60.0
    assert clock.now() == 60.0


@pytest.mark.asyncio
async def test_concurrent_acquire_is_async_safe() -> None:
    clock = FakeClock()
    limiter = ProviderRateLimiter(clock=clock.now, sleep=clock.sleep)
    limiter.configure(
        "openai", "default", RateLimitPolicy(requests_per_minute=1, tokens_per_minute=1000)
    )

    async def worker() -> float:
        return await limiter.acquire("openai", tokens=1)

    wait_one, wait_two = await asyncio.gather(worker(), worker())
    waits = sorted((wait_one, wait_two))

    assert waits[0] == 0.0
    assert waits[1] == 60.0


@pytest.mark.asyncio
async def test_bypass_requires_pre_call_throttle_or_explicit_denial() -> None:
    clock = FakeClock()
    limiter = ProviderRateLimiter(clock=clock.now, sleep=clock.sleep)
    limiter.configure(
        "openai", "default", RateLimitPolicy(requests_per_minute=1, tokens_per_minute=100)
    )

    _ = await limiter.acquire("openai", tokens=1)

    with pytest.raises(RateLimitExceeded) as error:
        _ = await limiter.acquire("openai", tokens=1, wait=False)

    assert error.value.wait_seconds == 60.0
