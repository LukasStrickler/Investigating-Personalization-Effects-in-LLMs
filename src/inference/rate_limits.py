from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None

    def __post_init__(self) -> None:
        if self.requests_per_minute is not None and self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive when provided")
        if self.tokens_per_minute is not None and self.tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive when provided")


class RateLimitExceeded(RuntimeError):
    provider: str
    profile: str
    wait_seconds: float

    def __init__(self, *, provider: str, profile: str, wait_seconds: float) -> None:
        self.provider = provider
        self.profile = profile
        self.wait_seconds = wait_seconds
        message = (
            f"Rate limit exceeded for provider={provider!r}, profile={profile!r}. "
            f"Retry in {wait_seconds:.3f}s"
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _BucketKey:
    provider: str
    profile: str


@dataclass(slots=True)
class _BucketState:
    lock: asyncio.Lock
    request_timestamps: deque[float]
    token_events: deque[tuple[float, int]]
    token_total: int = 0


class ProviderRateLimiter:
    def __init__(
        self,
        *,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
        window_seconds: float = 60.0,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._clock: Clock = clock or time.monotonic
        self._sleep: Sleep = sleep or asyncio.sleep
        self._window_seconds: float = window_seconds
        self._policies: dict[_BucketKey, RateLimitPolicy] = {}
        self._buckets: dict[_BucketKey, _BucketState] = {}

    def configure(self, provider: str, profile: str, policy: RateLimitPolicy) -> None:
        key = _BucketKey(provider=provider, profile=profile)
        self._policies[key] = policy
        _ = self._ensure_bucket(key)

    async def acquire(
        self,
        provider: str,
        *,
        profile: str = "default",
        tokens: int = 0,
        wait: bool = True,
    ) -> float:
        if tokens < 0:
            raise ValueError("tokens must be >= 0")

        key, policy = self._resolve_policy(provider=provider, profile=profile)
        token_limit = policy.tokens_per_minute
        if token_limit is not None and tokens > token_limit:
            raise ValueError("tokens exceeds tokens_per_minute and can never be scheduled")

        bucket = self._ensure_bucket(key)
        waited = 0.0

        while True:
            async with bucket.lock:
                now = self._clock()
                self._prune(bucket, now)
                wait_seconds = self._required_wait_seconds(
                    policy=policy, bucket=bucket, now=now, tokens=tokens
                )
                if wait_seconds <= 0:
                    bucket.request_timestamps.append(now)
                    if tokens > 0:
                        bucket.token_events.append((now, tokens))
                        bucket.token_total += tokens
                    return waited

            if not wait:
                raise RateLimitExceeded(
                    provider=key.provider, profile=key.profile, wait_seconds=wait_seconds
                )

            await self._sleep(wait_seconds)
            waited += wait_seconds

    def _resolve_policy(self, *, provider: str, profile: str) -> tuple[_BucketKey, RateLimitPolicy]:
        primary_key = _BucketKey(provider=provider, profile=profile)
        policy = self._policies.get(primary_key)
        if policy is not None:
            return primary_key, policy

        default_key = _BucketKey(provider=provider, profile="default")
        default_policy = self._policies.get(default_key)
        if default_policy is not None:
            return default_key, default_policy

        raise KeyError(
            f"No rate-limit policy configured for provider={provider!r}, profile={profile!r}"
        )

    def _ensure_bucket(self, key: _BucketKey) -> _BucketState:
        bucket = self._buckets.get(key)
        if bucket is not None:
            return bucket
        bucket = _BucketState(lock=asyncio.Lock(), request_timestamps=deque(), token_events=deque())
        self._buckets[key] = bucket
        return bucket

    def _prune(self, bucket: _BucketState, now: float) -> None:
        cutoff = now - self._window_seconds
        while bucket.request_timestamps and bucket.request_timestamps[0] <= cutoff:
            _ = bucket.request_timestamps.popleft()

        while bucket.token_events and bucket.token_events[0][0] <= cutoff:
            _, token_count = bucket.token_events.popleft()
            bucket.token_total -= token_count

    def _required_wait_seconds(
        self,
        *,
        policy: RateLimitPolicy,
        bucket: _BucketState,
        now: float,
        tokens: int,
    ) -> float:
        request_wait = self._request_wait_seconds(policy=policy, bucket=bucket, now=now)
        token_wait = self._token_wait_seconds(policy=policy, bucket=bucket, now=now, tokens=tokens)
        return max(request_wait, token_wait)

    def _request_wait_seconds(
        self, *, policy: RateLimitPolicy, bucket: _BucketState, now: float
    ) -> float:
        request_limit = policy.requests_per_minute
        if request_limit is None or len(bucket.request_timestamps) < request_limit:
            return 0.0

        earliest = bucket.request_timestamps[0]
        return max(0.0, (earliest + self._window_seconds) - now)

    def _token_wait_seconds(
        self,
        *,
        policy: RateLimitPolicy,
        bucket: _BucketState,
        now: float,
        tokens: int,
    ) -> float:
        token_limit = policy.tokens_per_minute
        if token_limit is None or tokens <= 0:
            return 0.0

        projected_tokens = bucket.token_total + tokens
        if projected_tokens <= token_limit:
            return 0.0

        deficit = projected_tokens - token_limit
        reclaimed = 0
        for timestamp, token_count in bucket.token_events:
            reclaimed += token_count
            if reclaimed >= deficit:
                return max(0.0, (timestamp + self._window_seconds) - now)

        return self._window_seconds
