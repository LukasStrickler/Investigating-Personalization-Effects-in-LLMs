from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from inference.client import InferenceRequest, InferenceResult
from inference.logging import LogEntry
from inference.retry import classify_error


class BatchCheckpointError(RuntimeError):
    pass


class SupportsInferenceClient(Protocol):
    async def complete(self, request: InferenceRequest) -> InferenceResult: ...


class SupportsInferenceLogger(Protocol):
    def write(self, entry: LogEntry) -> None: ...


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FATAL_FAILURE = "fatal_failure"
    RETRYABLE_FAILURE = "retryable_failure"


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    request_id: str
    status: CheckpointStatus
    timestamp: str
    error_message: str | None = None
    result: dict[str, Any] | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        if self.result is not None:
            payload["result"] = self.result
        return json.dumps(payload, separators=(",", ":"))


class BatchRunner:
    def __init__(
        self,
        *,
        client: SupportsInferenceClient,
        logger: SupportsInferenceLogger,
        checkpoint_path: str | Path,
    ) -> None:
        self._client = client
        self._logger = logger
        self._checkpoint_path = Path(checkpoint_path)

    async def run_batch(self, requests: AsyncIterator[InferenceRequest]) -> None:
        status_by_request = await self._load_checkpoint_state()

        index = 0
        async for request in requests:
            request_id = self._request_id(request=request, index=index)
            index += 1

            if status_by_request.get(request_id) in {
                CheckpointStatus.SUCCESS,
                CheckpointStatus.FATAL_FAILURE,
            }:
                continue

            await self._append_checkpoint(
                CheckpointEntry(
                    request_id=request_id,
                    status=CheckpointStatus.PENDING,
                    timestamp=_utc_timestamp(),
                )
            )
            status_by_request[request_id] = CheckpointStatus.PENDING

            try:
                result = await self._client.complete(request)
            except Exception as error:
                root_error = _root_error(error)
                category = classify_error(root_error)
                failure_status = (
                    CheckpointStatus.RETRYABLE_FAILURE
                    if category.is_retryable
                    else CheckpointStatus.FATAL_FAILURE
                )
                failure_entry = CheckpointEntry(
                    request_id=request_id,
                    status=failure_status,
                    timestamp=_utc_timestamp(),
                    error_message=str(root_error),
                )
                await self._append_checkpoint(failure_entry)
                status_by_request[request_id] = failure_status
                self._logger.write(
                    LogEntry(
                        provider="batch",
                        model=request.model_alias,
                        status="failure",
                        latency_ms=0.0,
                        error_type=type(root_error).__name__,
                        error_message=str(root_error),
                    )
                )
                continue

            success_entry = CheckpointEntry(
                request_id=request_id,
                status=CheckpointStatus.SUCCESS,
                timestamp=_utc_timestamp(),
                result=asdict(result),
            )
            await self._append_checkpoint(success_entry)
            status_by_request[request_id] = CheckpointStatus.SUCCESS
            self._logger.write(
                LogEntry(
                    provider=result.provider,
                    model=result.model,
                    status="success",
                    latency_ms=result.latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    retry_count=result.retry_count,
                )
            )

    def _request_id(self, request: InferenceRequest, *, index: int) -> str:
        dynamic_request_id = getattr(request, "request_id", None)
        if dynamic_request_id is not None:
            return str(dynamic_request_id)
        return str(index)

    async def _load_checkpoint_state(self) -> dict[str, CheckpointStatus]:
        return await asyncio.to_thread(self._load_checkpoint_state_sync)

    def _load_checkpoint_state_sync(self) -> dict[str, CheckpointStatus]:
        if not self._checkpoint_path.exists():
            return {}

        statuses: dict[str, CheckpointStatus] = {}
        with open(self._checkpoint_path, encoding="utf-8") as checkpoint_file:
            for line_number, raw_line in enumerate(checkpoint_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise BatchCheckpointError(
                        f"Corrupt checkpoint file at {self._checkpoint_path} line {line_number}. "
                        "Delete or repair the checkpoint file before resuming."
                    ) from error

                request_id = payload.get("request_id")
                status = payload.get("status")
                if not isinstance(request_id, str) or not isinstance(status, str):
                    raise BatchCheckpointError(
                        f"Invalid checkpoint entry at {self._checkpoint_path} line {line_number}. "
                        "Delete or repair the checkpoint file before resuming."
                    )

                try:
                    statuses[request_id] = CheckpointStatus(status)
                except ValueError as error:
                    raise BatchCheckpointError(
                        f"Unknown checkpoint status {status!r} at {self._checkpoint_path} line "
                        f"{line_number}. Delete or repair the checkpoint file before resuming."
                    ) from error

        return statuses

    async def _append_checkpoint(self, entry: CheckpointEntry) -> None:
        await asyncio.to_thread(self._append_checkpoint_sync, entry)

    def _append_checkpoint_sync(self, entry: CheckpointEntry) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._checkpoint_path, "a", encoding="utf-8") as checkpoint_file:
            checkpoint_file.write(entry.to_json())
            checkpoint_file.write("\n")
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())


def _root_error(error: Exception) -> Exception:
    cause = getattr(error, "__cause__", None)
    if isinstance(cause, Exception):
        return cause
    return error


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


__all__ = [
    "BatchCheckpointError",
    "BatchRunner",
    "CheckpointEntry",
    "CheckpointStatus",
]
