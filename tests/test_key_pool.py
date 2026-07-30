from __future__ import annotations

import pytest

from inference.config import resolve_api_keys
from inference.key_pool import ApiKeyPool, CooldownDefaults, parse_cooldown_seconds
from inference.types import ProviderConfig, ProviderName, RateLimit


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _provider(name: ProviderName = "openrouter", env: str = "OPENROUTER_API_KEY") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        api_key_env=env,
        rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
    )


class TestResolveApiKeys:
    def test_single_key_no_comma(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-solo")
        assert resolve_api_keys(_provider()) == ["sk-or-v1-solo"]

    def test_comma_parsing_strips_and_drops_empties(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", " a , b ,, c , ")
        assert resolve_api_keys(_provider()) == ["a", "b", "c"]

    def test_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            resolve_api_keys(_provider())

    def test_empty_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", " , ,")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            resolve_api_keys(_provider())

    def test_mock_returns_placeholder_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOCK_API_KEY", raising=False)
        assert resolve_api_keys(_provider(name="mock", env="MOCK_API_KEY")) == ["mock-key-for-mock"]


class TestApiKeyPool:
    def test_requires_at_least_one_key(self) -> None:
        with pytest.raises(ValueError):
            ApiKeyPool([])

    @pytest.mark.asyncio
    async def test_sticky_no_rotation_until_blocked(self) -> None:
        pool = ApiKeyPool(["a", "b"], clock=FakeClock())
        assert await pool.current_key() == "a"
        assert await pool.current_key() == "a"

    @pytest.mark.asyncio
    async def test_rotation_on_report_advances_to_next(self) -> None:
        pool = ApiKeyPool(["a", "b"], clock=FakeClock())
        outcome = await pool.report_rate_limited("a", 60.0)
        assert outcome.rotated is True
        assert outcome.all_exhausted is False
        assert await pool.current_key() == "b"

    @pytest.mark.asyncio
    async def test_stale_report_does_not_double_advance(self) -> None:
        clock = FakeClock()
        pool = ApiKeyPool(["a", "b", "c"], clock=clock)
        first = await pool.report_rate_limited("a", 100.0)
        assert first.rotated is True
        assert await pool.current_key() == "b"

        stale = await pool.report_rate_limited("a", 10.0)
        assert stale.rotated is False
        assert stale.all_exhausted is False
        assert await pool.current_key() == "b"

    @pytest.mark.asyncio
    async def test_stale_report_keeps_max_cooldown(self) -> None:
        clock = FakeClock()
        pool = ApiKeyPool(["a", "b"], clock=clock)
        await pool.report_rate_limited("a", 100.0)  # rotate to b
        await pool.report_rate_limited("a", 10.0)  # stale, must NOT shrink a's deadline
        outcome = await pool.report_rate_limited("b", 200.0)
        assert outcome.all_exhausted is True
        # min remaining is a's 100s, not the stale 10s
        assert outcome.wait_seconds == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_advance_skips_cooling_keys(self) -> None:
        clock = FakeClock()
        pool = ApiKeyPool(["a", "b", "c"], clock=clock)
        # b cools via a stale report (it is not the active key)
        await pool.report_rate_limited("b", 60.0)
        outcome = await pool.report_rate_limited("a", 60.0)
        assert outcome.rotated is True
        assert await pool.current_key() == "c"

    @pytest.mark.asyncio
    async def test_all_exhausted_returns_min_wait(self) -> None:
        clock = FakeClock()
        pool = ApiKeyPool(["a", "b"], clock=clock)
        await pool.report_rate_limited("a", 100.0)
        outcome = await pool.report_rate_limited("b", 50.0)
        assert outcome.rotated is False
        assert outcome.all_exhausted is True
        assert outcome.wait_seconds == pytest.approx(50.0)
        assert await pool.seconds_until_available() == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_current_key_advances_after_cooldown_expiry(self) -> None:
        clock = FakeClock()
        pool = ApiKeyPool(["a", "b"], clock=clock)
        await pool.report_rate_limited("a", 50.0)  # rotate to b
        await pool.report_rate_limited("b", 100.0)  # all exhausted, pointer stays on b
        clock.advance(60.0)  # a's cooldown expired, b still cooling
        assert await pool.current_key() == "a"
        assert await pool.seconds_until_available() == 0.0

    @pytest.mark.asyncio
    async def test_expired_cooldown_makes_key_reusable(self) -> None:
        clock = FakeClock()
        pool = ApiKeyPool(["a", "b"], clock=clock)
        await pool.report_rate_limited("a", 50.0)
        clock.advance(60.0)
        outcome = await pool.report_rate_limited("b", 50.0)
        assert outcome.rotated is True
        assert await pool.current_key() == "a"


class TestParseCooldownSeconds:
    NOW = 1_717_459_200.0  # 2024-06-04 00:00:00 UTC

    def test_per_day_until_utc_midnight(self) -> None:
        error = RuntimeError("429 Rate limit exceeded: free-models-per-day")
        now = self.NOW + 3600.0  # 01:00 UTC
        assert parse_cooldown_seconds(error, now_wall=now) == pytest.approx(82800.0)

    def test_per_hour(self) -> None:
        error = RuntimeError("429 rate limit exceeded: requests per hour")
        assert parse_cooldown_seconds(error, now_wall=self.NOW) == 3600.0

    def test_per_minute(self) -> None:
        error = RuntimeError("429 Rate limit exceeded: free-models-per-min-...")
        assert parse_cooldown_seconds(error, now_wall=self.NOW) == 65.0

    def test_unknown_429_uses_default(self) -> None:
        error = RuntimeError("429 too many requests")
        assert parse_cooldown_seconds(error, now_wall=self.NOW) == 420.0

    def test_reset_header_ms_epoch_wins_over_scope(self) -> None:
        reset_ms = int((self.NOW + 500.0) * 1000)
        error = RuntimeError(
            '429 free-models-per-day {"metadata":{"headers":'
            f'{{"X-RateLimit-Remaining":"0","X-RateLimit-Reset":"{reset_ms}"}}}}'
        )
        assert parse_cooldown_seconds(error, now_wall=self.NOW) == pytest.approx(500.0)

    def test_reset_header_seconds_epoch(self) -> None:
        reset_s = int(self.NOW + 300.0)
        error = RuntimeError(f"429 rate limit, X-RateLimit-Reset: {reset_s}")
        assert parse_cooldown_seconds(error, now_wall=self.NOW) == pytest.approx(300.0)

    def test_reset_header_capped(self) -> None:
        reset_ms = int((self.NOW + 100 * 86400.0) * 1000)
        error = RuntimeError(f'429 "X-RateLimit-Reset":"{reset_ms}"')
        assert parse_cooldown_seconds(error, now_wall=self.NOW) == CooldownDefaults().max_cap

    def test_reset_header_in_past_falls_back_to_scope(self) -> None:
        reset_ms = int((self.NOW - 100.0) * 1000)
        error = RuntimeError(f'429 per-hour "X-RateLimit-Reset":"{reset_ms}"')
        assert parse_cooldown_seconds(error, now_wall=self.NOW) == 3600.0

    def test_custom_defaults(self) -> None:
        defaults = CooldownDefaults(per_minute=5.0, per_hour=10.0, unknown=1.0)
        error = RuntimeError("429 rate limit per-minute")
        assert parse_cooldown_seconds(error, now_wall=self.NOW, defaults=defaults) == 5.0
