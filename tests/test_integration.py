from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from inference.client import InferenceRequest
from inference.types import (
    InferenceConfig,
    ModelAliasConfig,
    ProviderConfig,
    RateLimit,
    RetryConfig,
)


class ManualClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current += seconds


class SimulatedInterruption(BaseException):
    pass


def _build_config(
    log_path: Path,
    *,
    checkpoint_path: Path,
    requests_per_minute: int,
) -> InferenceConfig:
    return InferenceConfig(
        providers={
            "openai": ProviderConfig(
                name="openai",
                api_key_env="OPENAI_API_KEY",
                rate_limit=RateLimit(
                    requests_per_minute=requests_per_minute,
                    tokens_per_minute=0,
                ),
            ),
            "mock": ProviderConfig(
                name="mock",
                api_key_env="MOCK_API_KEY",
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
        },
        default_provider="openai",
        model_aliases={
            "integration-openai": ModelAliasConfig(
                alias="integration-openai",
                provider="openai",
                model="gpt-4o-mini",
            ),
            "integration-mock": ModelAliasConfig(
                alias="integration-mock",
                provider="mock",
                model="mock-model-v1",
            ),
        },
        default_retry=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.01),
        log_path=str(log_path),
        checkpoint_path=str(checkpoint_path),
    )


async def _iter_requests(requests: list[InferenceRequest]) -> AsyncIterator[InferenceRequest]:
    for request in requests:
        yield request


def _latest_statuses(checkpoint_path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        statuses[payload["request_id"]] = payload["status"]
    return statuses


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.mark.asyncio
async def test_integration_pipeline_handles_success_timeout_429_and_auth_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.batch import BatchRunner
    from inference.client import UnifiedInferenceClient
    from inference.logging import InferenceLogger
    from inference.providers import LiteLLMProviderAdapter
    from inference.rate_limits import ProviderRateLimiter

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    perf_counter_state = {"now": 0.0}

    def fake_perf_counter() -> float:
        perf_counter_state["now"] += 0.001
        return perf_counter_state["now"]

    monkeypatch.setattr("inference.client.time.perf_counter", fake_perf_counter)
    monkeypatch.setattr("inference.client.calculate_backoff", lambda *_args, **_kwargs: 0.0)

    attempts: dict[str, int] = {}

    async def scripted_completion(**kwargs: Any) -> dict[str, Any]:
        prompt = str(kwargs["messages"][0]["content"])
        attempts[prompt] = attempts.get(prompt, 0) + 1

        if prompt == "timeout" and attempts[prompt] == 1:
            raise TimeoutError("timeout while calling provider")
        if prompt == "rate-limit" and attempts[prompt] == 1:
            raise RuntimeError("429 rate limit")
        if prompt == "auth-failure":
            raise RuntimeError("401 unauthorized")

        return {
            "choices": [{"message": {"content": f"ok:{prompt}"}}],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        }

    request_log_path = tmp_path / "request-log.jsonl"
    batch_log_path = tmp_path / "batch-log.jsonl"
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    config = _build_config(
        request_log_path,
        checkpoint_path=checkpoint_path,
        requests_per_minute=1,
    )
    clock = ManualClock()
    limiter = ProviderRateLimiter(clock=clock.now, sleep=clock.sleep, window_seconds=5.0)
    client = UnifiedInferenceClient(
        config=config,
        adapter=LiteLLMProviderAdapter(completion_callable=scripted_completion),
        limiter=limiter,
        sleep=clock.sleep,
    )
    runner = BatchRunner(
        client=client,
        logger=InferenceLogger(batch_log_path),
        checkpoint_path=checkpoint_path,
    )

    requests = [
        InferenceRequest(model_alias="integration-openai", prompt="happy"),
        InferenceRequest(model_alias="integration-openai", prompt="timeout"),
        InferenceRequest(model_alias="integration-openai", prompt="rate-limit"),
        InferenceRequest(model_alias="integration-openai", prompt="auth-failure"),
    ]
    await runner.run_batch(_iter_requests(requests))

    assert attempts == {
        "happy": 1,
        "timeout": 2,
        "rate-limit": 2,
        "auth-failure": 1,
    }
    assert any(wait > 0.0 for wait in clock.sleep_calls)
    assert _latest_statuses(checkpoint_path) == {
        "0": "success",
        "1": "success",
        "2": "success",
        "3": "fatal_failure",
    }

    request_logs = _read_jsonl(request_log_path)
    assert [entry["status"] for entry in request_logs] == [
        "success",
        "success",
        "success",
        "failure",
    ]
    assert request_logs[1]["retry_count"] == 1
    assert request_logs[2]["retry_count"] == 1
    assert request_logs[3]["error_type"] == "RuntimeError"

    batch_logs = _read_jsonl(batch_log_path)
    assert [entry["status"] for entry in batch_logs] == ["success", "success", "success", "failure"]


@pytest.mark.asyncio
async def test_integration_resume_after_interruption_replays_only_incomplete_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.batch import BatchRunner
    from inference.client import UnifiedInferenceClient
    from inference.logging import InferenceLogger
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    checkpoint_path = tmp_path / "checkpoint.jsonl"
    request_log_path = tmp_path / "request-log.jsonl"
    batch_log_path = tmp_path / "batch-log.jsonl"
    config = _build_config(
        request_log_path,
        checkpoint_path=checkpoint_path,
        requests_per_minute=0,
    )

    first_run_calls: list[str] = []

    async def interrupting_completion(**kwargs: Any) -> dict[str, Any]:
        prompt = str(kwargs["messages"][0]["content"])
        first_run_calls.append(prompt)
        if prompt == "resume-1":
            raise SimulatedInterruption("stop run")
        return {
            "choices": [{"message": {"content": f"ok:{prompt}"}}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    first_runner = BatchRunner(
        client=UnifiedInferenceClient(
            config=config,
            adapter=LiteLLMProviderAdapter(completion_callable=interrupting_completion),
        ),
        logger=InferenceLogger(batch_log_path),
        checkpoint_path=checkpoint_path,
    )

    requests = [
        InferenceRequest(model_alias="integration-openai", prompt="resume-0"),
        InferenceRequest(model_alias="integration-openai", prompt="resume-1"),
        InferenceRequest(model_alias="integration-openai", prompt="resume-2"),
    ]
    with pytest.raises(SimulatedInterruption):
        await first_runner.run_batch(_iter_requests(requests))

    assert first_run_calls == ["resume-0", "resume-1"]
    assert _latest_statuses(checkpoint_path) == {
        "0": "success",
        "1": "pending",
    }

    second_run_calls: list[str] = []

    async def resumed_completion(**kwargs: Any) -> dict[str, Any]:
        prompt = str(kwargs["messages"][0]["content"])
        second_run_calls.append(prompt)
        return {
            "choices": [{"message": {"content": f"ok:{prompt}"}}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    second_runner = BatchRunner(
        client=UnifiedInferenceClient(
            config=config,
            adapter=LiteLLMProviderAdapter(completion_callable=resumed_completion),
        ),
        logger=InferenceLogger(batch_log_path),
        checkpoint_path=checkpoint_path,
    )
    await second_runner.run_batch(_iter_requests(requests))

    assert second_run_calls == ["resume-1", "resume-2"]
    assert _latest_statuses(checkpoint_path) == {
        "0": "success",
        "1": "success",
        "2": "success",
    }


@pytest.mark.asyncio
async def test_no_network_mock_provider_flow_works_without_api_keys_or_litellm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.delenv("MOCK_API_KEY", raising=False)

    def fail_litellm_load() -> Any:
        raise AssertionError("litellm completion should not load for mock provider")

    def fail_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access should not happen in mocked integration tests")

    monkeypatch.setattr("inference.providers._get_litellm_acompletion", fail_litellm_load)
    monkeypatch.setattr("socket.create_connection", fail_network)

    checkpoint_path = tmp_path / "checkpoint.jsonl"
    config = _build_config(
        tmp_path / "request-log.jsonl",
        checkpoint_path=checkpoint_path,
        requests_per_minute=0,
    )
    client = UnifiedInferenceClient(config=config, adapter=LiteLLMProviderAdapter())

    result = await client.complete(
        InferenceRequest(model_alias="integration-mock", prompt="no-network")
    )

    assert result.provider == "mock"
    assert result.content == "mock-response:no-network"
