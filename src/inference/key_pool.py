"""Sticky API key pool with cooldown-based rotation for rate-limited providers.

Multiple keys are supplied via a comma-separated env var (see
``inference.config.resolve_api_keys``). One key is active at a time; when a
request hits a key-scoped rate limit (e.g. OpenRouter's daily free-request
quota), the key is put on cooldown and the pool advances to the next
available key. The pool is sticky: it never advances on its own while the
active key is healthy.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CooldownDefaults:
    """Fallback cooldown durations (seconds) when no reset timestamp is parseable."""

    per_minute: float = 65.0
    per_hour: float = 3600.0
    unknown: float = 420.0
    # Upper bound so a corrupt reset header cannot park a key forever.
    max_cap: float = 26 * 3600.0


# OpenRouter embeds its JSON error body (incl. rate limit headers) in the
# exception message raised by LiteLLM, so all parsing is regex-on-string.
_RESET_HEADER_RE = re.compile(
    r"X-RateLimit-Reset[\"']?\s*[:=]\s*[\"']?(\d{10,16})", re.IGNORECASE
)
_PER_DAY_RE = re.compile(r"free-models-per-day|per[-\s]day|daily", re.IGNORECASE)
_PER_HOUR_RE = re.compile(r"per[-\s]hour|hourly", re.IGNORECASE)
_PER_MINUTE_RE = re.compile(r"per[-\s]min(?:ute)?", re.IGNORECASE)


def _seconds_to_utc_midnight(now_wall: float) -> float:
    now = datetime.fromtimestamp(now_wall, tz=timezone.utc)
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1.0, (next_midnight - now).total_seconds())


def parse_cooldown_seconds(
    error: Exception,
    *,
    now_wall: float | None = None,
    defaults: CooldownDefaults | None = None,
) -> float:
    """Derive a key cooldown duration from an OpenRouter-style rate-limit error.

    Priority: absolute ``X-RateLimit-Reset`` timestamp (ms or s epoch) embedded
    in the error body, then scope keywords (per-day resets at UTC midnight,
    per-hour, per-minute), then a generic default.
    """
    message = str(error)
    now = time.time() if now_wall is None else now_wall
    cooldowns = defaults or CooldownDefaults()

    match = _RESET_HEADER_RE.search(message)
    if match is not None:
        reset = float(match.group(1))
        if reset >= 1e12:  # millisecond epoch
            reset /= 1000.0
        remaining = reset - now
        if remaining > 0:
            return min(remaining, cooldowns.max_cap)

    if _PER_DAY_RE.search(message):
        return min(_seconds_to_utc_midnight(now), cooldowns.max_cap)
    if _PER_HOUR_RE.search(message):
        return cooldowns.per_hour
    if _PER_MINUTE_RE.search(message):
        return cooldowns.per_minute
    return cooldowns.unknown


@dataclass(slots=True)
class _KeyState:
    key: str
    cooldown_until: float = 0.0  # monotonic deadline; 0 == available


@dataclass(frozen=True, slots=True)
class RotationOutcome:
    rotated: bool
    """True if the pointer advanced to a fresh key."""
    all_exhausted: bool
    """True if every key in the pool is cooling down."""
    wait_seconds: float
    """Seconds until the earliest cooldown expires; >0 only when all_exhausted."""


class ApiKeyPool:
    """Sticky pool: one active key until it is reported rate-limited."""

    def __init__(self, keys: list[str], *, clock: Clock | None = None) -> None:
        if not keys:
            raise ValueError("ApiKeyPool requires at least one key")
        self._clock = clock or time.monotonic
        self._states = [_KeyState(key=key) for key in keys]
        self._current = 0
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._states)

    @property
    def primary_key(self) -> str:
        """The first configured key — used for paid (non-rotating) traffic."""
        return self._states[0].key

    async def current_key(self) -> str:
        """Return the active key, advancing past it if it is cooling and another is free."""
        async with self._lock:
            now = self._clock()
            if self._states[self._current].cooldown_until > now:
                self._advance_locked(now)
            return self._states[self._current].key

    async def report_rate_limited(self, key: str, cooldown_seconds: float) -> RotationOutcome:
        """Put ``key`` on cooldown; advance only if it is still the active key.

        Stale-key guard: a late report from an already-rotated-away key records
        the cooldown (it is genuinely limited) but does not advance the pointer
        again, so concurrent failures on the same key cause a single rotation.
        """
        async with self._lock:
            now = self._clock()
            deadline = now + max(0.0, cooldown_seconds)
            for state in self._states:
                if state.key == key:
                    state.cooldown_until = max(state.cooldown_until, deadline)
                    break
            if self._states[self._current].key != key:
                return RotationOutcome(rotated=False, all_exhausted=False, wait_seconds=0.0)
            return self._advance_locked(now)

    async def seconds_until_available(self) -> float:
        """Min remaining cooldown across all keys; 0.0 if any key is available."""
        async with self._lock:
            return self._min_wait_locked(self._clock())

    def _advance_locked(self, now: float) -> RotationOutcome:
        count = len(self._states)
        for offset in range(1, count + 1):
            candidate = (self._current + offset) % count
            if self._states[candidate].cooldown_until <= now:
                self._current = candidate
                return RotationOutcome(rotated=True, all_exhausted=False, wait_seconds=0.0)
        return RotationOutcome(
            rotated=False, all_exhausted=True, wait_seconds=self._min_wait_locked(now)
        )

    def _min_wait_locked(self, now: float) -> float:
        return min(max(0.0, state.cooldown_until - now) for state in self._states)


__all__ = [
    "ApiKeyPool",
    "CooldownDefaults",
    "RotationOutcome",
    "parse_cooldown_seconds",
]
