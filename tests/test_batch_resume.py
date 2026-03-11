from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.client import InferenceRequest, InferenceResult


class LoggerSpy:
    def __init__(self) -> None:
        self.entries: list[object] = []

    def write(self, entry: object) -> None:
        self.entries.append(entry)


class FakeClient:
    def __init__(self, outcomes: dict[str, object] | None = None) -> None:
        self._outcomes = outcomes or {}
        self.calls: list[str] = []

    async def complete(self, request: InferenceRequest) -> InferenceResult:
        self.calls.append(request.model_alias)
        outcome = self._outcomes.get(request.model_alias)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, InferenceResult):
            return outcome
        return InferenceResult(
            model_alias=request.model_alias,
            provider="mock",
            model="mock-model",
            content=f"ok:{request.model_alias}",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_ms=1.0,
            retry_count=0,
        )


class SimulatedInterruption(BaseException):
    pass


async def _iter_requests(requests: list[InferenceRequest]):
    for request in requests:
        yield request


def _latest_statuses(checkpoint_path: Path) -> dict[str, str]:
    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    statuses: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        statuses[payload["request_id"]] = payload["status"]
    return statuses


@pytest.mark.asyncio
async def test_interrupted_batch_resumes_without_replaying_completed_items(tmp_path: Path) -> None:
    from inference.batch import BatchRunner

    checkpoint_path = tmp_path / "checkpoint.jsonl"
    requests = [
        InferenceRequest(model_alias="req-0", prompt="p0"),
        InferenceRequest(model_alias="req-1", prompt="p1"),
        InferenceRequest(model_alias="req-2", prompt="p2"),
        InferenceRequest(model_alias="req-3", prompt="p3"),
    ]

    first_run_client = FakeClient(outcomes={"req-2": SimulatedInterruption("stop")})
    first_logger = LoggerSpy()
    runner = BatchRunner(
        client=first_run_client,
        logger=first_logger,
        checkpoint_path=checkpoint_path,
    )

    with pytest.raises(SimulatedInterruption):
        await runner.run_batch(_iter_requests(requests))

    assert first_run_client.calls == ["req-0", "req-1", "req-2"]

    resumed_client = FakeClient()
    resumed_logger = LoggerSpy()
    resumed_runner = BatchRunner(
        client=resumed_client,
        logger=resumed_logger,
        checkpoint_path=checkpoint_path,
    )
    await resumed_runner.run_batch(_iter_requests(requests))

    assert resumed_client.calls == ["req-2", "req-3"]
    assert _latest_statuses(checkpoint_path) == {
        "0": "success",
        "1": "success",
        "2": "success",
        "3": "success",
    }


@pytest.mark.asyncio
async def test_checkpoint_records_retryable_and_fatal_failures(tmp_path: Path) -> None:
    from inference.batch import BatchRunner

    checkpoint_path = tmp_path / "checkpoint.jsonl"
    requests = [
        InferenceRequest(model_alias="retryable", prompt="p0"),
        InferenceRequest(model_alias="fatal", prompt="p1"),
        InferenceRequest(model_alias="success", prompt="p2"),
    ]
    client = FakeClient(
        outcomes={
            "retryable": RuntimeError("429 rate limit"),
            "fatal": ValueError("400 invalid request"),
        }
    )
    logger = LoggerSpy()

    runner = BatchRunner(client=client, logger=logger, checkpoint_path=checkpoint_path)
    await runner.run_batch(_iter_requests(requests))

    assert _latest_statuses(checkpoint_path) == {
        "0": "retryable_failure",
        "1": "fatal_failure",
        "2": "success",
    }
    assert len(logger.entries) == 3


@pytest.mark.asyncio
async def test_corrupt_checkpoint_raises_clear_error(tmp_path: Path) -> None:
    from inference.batch import BatchCheckpointError, BatchRunner

    checkpoint_path = tmp_path / "checkpoint.jsonl"
    checkpoint_path.write_text("this is not json\n", encoding="utf-8")

    runner = BatchRunner(client=FakeClient(), logger=LoggerSpy(), checkpoint_path=checkpoint_path)

    with pytest.raises(BatchCheckpointError, match="Delete or repair"):
        await runner.run_batch(_iter_requests([InferenceRequest(model_alias="req", prompt="p")]))
