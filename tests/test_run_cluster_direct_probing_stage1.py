"""Tests for scripts/run_cluster_direct_probing_stage1.py and vllm_matrix."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from inference.experiments.persistence import MatrixCSVWriter
from inference.experiments.vllm_matrix import (
    DEMO_PROMPTS,
    CircuitTrippedError,
    ConnectionCircuitBreaker,
    MatrixColumnRun,
    is_connection_error,
    probe_openai_compatible_server,
    run_matrix_column,
)
from inference.providers import ProviderRequest, ProviderResponse

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "inference.vllm.example.yaml"
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_cluster_direct_probing_stage1.py"


def _load_cli():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("run_cluster_direct_probing_stage1", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_cli()


def test_help_exits_zero_and_lists_config(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        rv.main(["--help"])
    assert exc_info.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_missing_config_returns_nonzero(tmp_path: Path) -> None:
    code = rv.main(
        [
            "--config",
            str(tmp_path / "does-not-exist.yaml"),
            "--model-alias",
            "mock-test",
            "--prompts-source",
            "demo",
            "--limit",
            "1",
            "--csv-path",
            str(tmp_path / "matrix.csv"),
        ]
    )
    assert code == 2


def test_unknown_alias_returns_nonzero(tmp_path: Path) -> None:
    code = rv.main(
        [
            "--config",
            str(_CONFIG_PATH),
            "--model-alias",
            "no-such-alias",
            "--prompts-source",
            "demo",
            "--limit",
            "1",
            "--csv-path",
            str(tmp_path / "matrix.csv"),
        ]
    )
    assert code == 2


@pytest.mark.asyncio
async def test_mock_demo_run_fills_one_cell_then_resume_is_noop(tmp_path: Path) -> None:
    csv_path = tmp_path / "matrix.csv"
    run = MatrixColumnRun(
        config_path=_CONFIG_PATH,
        model_alias="mock-test",
        columns=["mock-test"],
        experiment_name="test",
        csv_path=csv_path,
        prompts_source="demo",
        limit=1,
        sample_per_group=10_000,
        circuit_breaker_threshold=5,
        skip_probe=True,
    )

    assert await run_matrix_column(run) == 0
    text = csv_path.read_text(encoding="utf-8")
    assert "success" in text and "mock-response" in text

    before = csv_path.read_bytes()
    assert await run_matrix_column(run) == 0
    assert csv_path.read_bytes() == before


@pytest.mark.asyncio
async def test_completion_marker_written_on_clean_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "matrix.csv"
    marker = Path(f"{csv_path}.mock-test.complete")
    run = MatrixColumnRun(
        config_path=_CONFIG_PATH,
        model_alias="mock-test",
        columns=["mock-test"],
        experiment_name="test",
        csv_path=csv_path,
        prompts_source="demo",
        limit=1,
        sample_per_group=10_000,
        circuit_breaker_threshold=5,
        skip_probe=True,
    )
    assert await run_matrix_column(run) == 0
    assert marker.exists()
    assert marker.read_text(encoding="utf-8").strip() == "complete"


@pytest.mark.asyncio
async def test_pinned_csv_with_disjoint_prompts_is_refused(tmp_path: Path) -> None:
    csv_path = tmp_path / "matrix.csv"
    other_prompts = [
        {
            "messages": [{"role": "user", "content": "a completely unrelated prompt"}],
            "metadata": {"id": "x"},
        },
    ]
    MatrixCSVWriter(csv_path, ["mock-test"]).initialize(other_prompts)

    run = MatrixColumnRun(
        config_path=_CONFIG_PATH,
        model_alias="mock-test",
        columns=["mock-test"],
        experiment_name="test",
        csv_path=csv_path,
        prompts_source="demo",
        limit=1,
        sample_per_group=10_000,
        circuit_breaker_threshold=5,
        skip_probe=True,
    )
    assert await run_matrix_column(run) == 6


@pytest.mark.asyncio
async def test_pinned_csv_with_subset_prompts_is_refused(tmp_path: Path) -> None:
    csv_path = tmp_path / "matrix.csv"
    MatrixCSVWriter(csv_path, ["mock-test"]).initialize(list(DEMO_PROMPTS))

    run = MatrixColumnRun(
        config_path=_CONFIG_PATH,
        model_alias="mock-test",
        columns=["mock-test"],
        experiment_name="test",
        csv_path=csv_path,
        prompts_source="demo",
        limit=1,
        sample_per_group=10_000,
        circuit_breaker_threshold=5,
        skip_probe=True,
    )
    assert await run_matrix_column(run) == 6


def test_probe_endpoint_reports_unreachable() -> None:
    ok, message = probe_openai_compatible_server(
        "http://127.0.0.1:1", served_model="m", timeout=0.1
    )
    assert not ok
    assert "cannot reach" in message.lower()


def test_is_connection_error_classifies_transport_failures() -> None:
    assert is_connection_error(ConnectionError("Connection refused"))
    assert is_connection_error(TimeoutError("timed out"))

    class APIConnectionError(Exception):
        pass

    assert is_connection_error(APIConnectionError("boom"))
    wrapped = RuntimeError("call failed")
    wrapped.__cause__ = ConnectionError("All connection attempts failed")
    assert is_connection_error(wrapped)
    assert not is_connection_error(ValueError("bad input"))


class _FakeAdapter:
    def __init__(self, error: Exception, *, heal_after: int | None = None) -> None:
        self._error = error
        self._heal_after = heal_after
        self.calls = 0

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        if self._heal_after is not None and self.calls > self._heal_after:
            return ProviderResponse(content="ok")
        raise self._error


def _req() -> ProviderRequest:
    return ProviderRequest(provider="vllm", model="m", prompt="hi")


def test_circuit_breaker_trips_after_threshold_then_fails_fast() -> None:
    adapter = _FakeAdapter(ConnectionError("Connection refused"))
    breaker = ConnectionCircuitBreaker(adapter, threshold=3)

    for _ in range(3):
        with pytest.raises(ConnectionError):
            asyncio.run(breaker.complete(_req()))
    assert breaker.tripped

    calls_at_trip = adapter.calls
    with pytest.raises(CircuitTrippedError):
        asyncio.run(breaker.complete(_req()))
    assert adapter.calls == calls_at_trip


def test_circuit_breaker_resets_on_success() -> None:
    adapter = _FakeAdapter(ConnectionError("Connection refused"), heal_after=2)
    breaker = ConnectionCircuitBreaker(adapter, threshold=3)

    for _ in range(2):
        with pytest.raises(ConnectionError):
            asyncio.run(breaker.complete(_req()))
    result = asyncio.run(breaker.complete(_req()))
    assert result.content == "ok"
    assert not breaker.tripped
