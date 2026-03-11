from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from inference.client import InferenceRequest
from inference.experiments.runner import ExperimentRunner
from inference.experiments.types import (
    ExperimentConfig,
    ExperimentRetryOptions,
    ExperimentSchedulingOptions,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, request: InferenceRequest) -> Any:
        self.calls.append((request.prompt, request.model_alias))
        if request.prompt == "prompt-2" and request.model_alias == "alias-b":
            await asyncio.sleep(0.03)
        return SimpleNamespace(content=f"{request.model_alias}:{request.prompt}")


@dataclass(slots=True)
class RetryAfterError(Exception):
    retry_after_seconds: float

    def __str__(self) -> str:
        return f"retry-after={self.retry_after_seconds}"


class ScriptedClient:
    def __init__(self, script: dict[tuple[str, str], list[object]]) -> None:
        self._script = {key: list(values) for key, values in script.items()}
        self.calls: dict[tuple[str, str], int] = dict.fromkeys(script, 0)

    async def complete(self, request: InferenceRequest) -> Any:
        key = (request.prompt, request.model_alias)
        self.calls[key] = self.calls.get(key, 0) + 1
        plan = self._script[key]
        outcome = plan.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(content=str(outcome))


def _status_row(result: Any) -> dict[str, str]:
    first_row = result.dataframe.iloc[0].to_dict()
    return {
        alias: cell_payload["status"]
        for alias, cell_payload in first_row.items()
        if alias not in {"prompt", "prompt_id"}
    }


@pytest.mark.asyncio
async def test_run_executes_full_prompt_alias_matrix_and_waits_for_all_cells() -> None:
    client = FakeClient()
    runner = ExperimentRunner(client=client)  # type: ignore[arg-type]
    config = ExperimentConfig(
        experiment_name="matrix-run",
        model_aliases=["alias-a", "alias-b"],
        prompts=["prompt-1", "prompt-2"],
    )

    started = time.perf_counter()
    result = await runner.run(config)
    elapsed = time.perf_counter() - started

    assert len(client.calls) == 4
    assert set(client.calls) == {
        ("prompt-1", "alias-a"),
        ("prompt-1", "alias-b"),
        ("prompt-2", "alias-a"),
        ("prompt-2", "alias-b"),
    }
    assert elapsed >= 0.02
    assert result.summary.total_cells == 4
    assert result.summary.completed_cells == 4
    assert result.summary.failed_cells == 0
    assert result.summary.rate_limited_cells == 0
    assert isinstance(result.dataframe, pd.DataFrame)
    assert result.csv_name == result.csv_path.name
    assert set(result.dataframe.columns) == {"prompt_id", "prompt", "alias-a", "alias-b"}


@pytest.mark.asyncio
async def test_run_uses_retry_resolution_for_short_and_long_retry_after() -> None:
    script = {
        ("prompt", "short-wait"): [RetryAfterError(0.0), "ok-short"],
        ("prompt", "long-wait"): [RetryAfterError(7200.0)],
        ("prompt", "fails"): [RuntimeError("network timeout"), RuntimeError("network timeout")],
    }
    client = ScriptedClient(script)
    runner = ExperimentRunner(client=client)  # type: ignore[arg-type]
    config = ExperimentConfig(
        experiment_name="retry-semantics",
        model_aliases=["short-wait", "long-wait", "fails"],
        prompts=["prompt"],
        retry=ExperimentRetryOptions(max_retries=2, base_delay=0.001, max_delay=0.001),
        scheduling=ExperimentSchedulingOptions(max_retry_after_wait_seconds=3600.0),
    )

    result = await runner.run(config)
    row_statuses = _status_row(result)

    assert client.calls[("prompt", "short-wait")] == 2
    assert client.calls[("prompt", "long-wait")] == 1
    assert client.calls[("prompt", "fails")] == 2
    assert row_statuses == {
        "short-wait": "success",
        "long-wait": "rate_limited",
        "fails": "failed",
    }
    assert result.summary.total_cells == 3
    assert result.summary.completed_cells == 3
    assert result.summary.failed_cells == 1
    assert result.summary.rate_limited_cells == 1


@pytest.mark.asyncio
async def test_types_experiment_runner_facade_delegates_to_concrete_runner() -> None:
    from inference.experiments.types import ExperimentRunner as FacadeRunner

    client = ScriptedClient({("prompt", "alias"): ["ok"]})
    runner = FacadeRunner(client=client)  # type: ignore[arg-type]
    config = ExperimentConfig(experiment_name="facade", model_aliases=["alias"], prompts=["prompt"])

    result = await runner.run(config)
    row_statuses = _status_row(result)

    assert row_statuses == {"alias": "success"}
    assert result.summary.total_cells == 1


@pytest.mark.asyncio
async def test_run_returns_dataframe_matching_csv_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = ScriptedClient(
        {
            ("prompt-1", "alias-a"): ["ok-a"],
            ("prompt-1", "alias-b"): [RuntimeError("provider down")],
        }
    )
    runner = ExperimentRunner(client=client)  # type: ignore[arg-type]
    config = ExperimentConfig(
        experiment_name="dataframe-materialization",
        model_aliases=["alias-a", "alias-b"],
        prompts=["prompt-1"],
    )

    result = await runner.run(config)

    assert isinstance(result.dataframe, pd.DataFrame)
    assert result.csv_path.exists()
    assert result.csv_name == result.csv_path.name
    assert result.dataframe.to_dict(orient="records") == [
        {
            "prompt_id": result.dataframe.iloc[0]["prompt_id"],
            "prompt": "prompt-1",
            "alias-a": {"status": "success", "response": "ok-a", "error_message": None},
            "alias-b": {
                "status": "failed",
                "response": None,
                "error_message": "provider down",
            },
        }
    ]


# === Resume Tests ===


@pytest.mark.asyncio
async def test_resume_skips_success_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume mode should skip cells already marked as SUCCESS in existing CSV."""
    monkeypatch.chdir(tmp_path)

    # First run: create CSV with success and failed cells
    client1 = ScriptedClient({
        ("prompt-1", "alias-a"): ["ok-a"],
        ("prompt-1", "alias-b"): [RuntimeError("fail-b")],
    })
    runner1 = ExperimentRunner(client=client1)  # type: ignore[arg-type]
    config1 = ExperimentConfig(
        experiment_name="resume-test",
        model_aliases=["alias-a", "alias-b"],
        prompts=["prompt-1"],
        retry=ExperimentRetryOptions(max_retries=1, base_delay=0.001, max_delay=0.001),
    )
    result1 = await runner1.run(config1)
    csv_path = result1.csv_path

    # Verify first run results
    row1 = _status_row(result1)
    assert row1 == {"alias-a": "success", "alias-b": "failed"}

    # Second run: resume with fixed client (only failed cell should be called)
    client2 = ScriptedClient({
        ("prompt-1", "alias-b"): ["ok-b-fixed"],
    })
    runner2 = ExperimentRunner(client=client2)  # type: ignore[arg-type]
    config2 = ExperimentConfig(
        experiment_name="resume-test",
        model_aliases=["alias-a", "alias-b"],
        prompts=["prompt-1"],
        resume_from_existing_csv=True,
        existing_csv_path=csv_path,
    )
    result2 = await runner2.run(config2)

    # Verify: alias-a was NOT called again (SUCCESS was skipped)
    assert ("prompt-1", "alias-a") not in client2.calls
    assert client2.calls[("prompt-1", "alias-b")] == 1

    # Verify same CSV file was updated
    assert result2.csv_path == csv_path

    # Verify final state shows both successful
    row2 = _status_row(result2)
    assert row2 == {"alias-a": "success", "alias-b": "success"}


@pytest.mark.asyncio
async def test_resume_retries_rate_limited_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume mode should retry RATE_LIMITED cells."""
    monkeypatch.chdir(tmp_path)

    # First run: create CSV with rate_limited cell
    client1 = ScriptedClient({
        ("prompt-1", "alias-a"): ["ok-a"],
        ("prompt-1", "alias-b"): [RetryAfterError(7200.0)],  # Long wait = rate_limited
    })
    runner1 = ExperimentRunner(client=client1)  # type: ignore[arg-type]
    config1 = ExperimentConfig(
        experiment_name="rate-limit-resume",
        model_aliases=["alias-a", "alias-b"],
        prompts=["prompt-1"],
        scheduling=ExperimentSchedulingOptions(max_retry_after_wait_seconds=3600.0),
    )
    result1 = await runner1.run(config1)
    csv_path = result1.csv_path

    # Verify first run results
    row1 = _status_row(result1)
    assert row1 == {"alias-a": "success", "alias-b": "rate_limited"}

    # Second run: resume with fixed client
    client2 = ScriptedClient({
        ("prompt-1", "alias-b"): ["ok-b-recovered"],
    })
    runner2 = ExperimentRunner(client=client2)  # type: ignore[arg-type]
    config2 = ExperimentConfig(
        experiment_name="rate-limit-resume",
        model_aliases=["alias-a", "alias-b"],
        prompts=["prompt-1"],
        resume_from_existing_csv=True,
        existing_csv_path=csv_path,
    )
    result2 = await runner2.run(config2)

    # Verify: alias-a was NOT called (SUCCESS skipped), alias-b WAS retried
    assert ("prompt-1", "alias-a") not in client2.calls
    assert client2.calls[("prompt-1", "alias-b")] == 1

    # Verify final state shows both successful
    row2 = _status_row(result2)
    assert row2 == {"alias-a": "success", "alias-b": "success"}


def test_empty_prompts_raises_value_error() -> None:
    """ExperimentConfig must reject empty prompts list (1..N constraint)."""
    with pytest.raises(ValueError, match="prompts must contain at least one prompt"):
        ExperimentConfig(
            experiment_name="empty-prompts",
            model_aliases=["alias-a"],
            prompts=[],  # Should raise
        )


def test_empty_model_aliases_raises_value_error() -> None:
    """ExperimentConfig must reject empty model_aliases list."""
    with pytest.raises(ValueError, match="model_aliases must contain at least one"):
        ExperimentConfig(
            experiment_name="empty-aliases",
            model_aliases=[],  # Should raise
            prompts=["prompt-1"],
        )


def test_empty_experiment_name_raises_value_error() -> None:
    """ExperimentConfig must reject empty experiment_name."""
    with pytest.raises(ValueError, match="experiment_name must be non-empty"):
        ExperimentConfig(
            experiment_name="",  # Should raise
            model_aliases=["alias-a"],
            prompts=["prompt-1"],
        )

