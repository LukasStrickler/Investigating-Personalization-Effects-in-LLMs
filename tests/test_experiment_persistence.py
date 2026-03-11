from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from inference.client import InferenceRequest
from inference.experiments.csv_schema import (
    CellStatus,
    MatrixCell,
    canonical_prompt_spec,
    compute_prompt_id,
    csv_writer_kwargs,
    serialize_prompt_content,
)
from inference.experiments.dataframe import (
    build_dataframe_from_csv,
    filter_experiment_dataframe,
    to_analysis_dataframe,
)
from inference.experiments.persistence import MatrixCSVWriter
from inference.experiments.runner import ExperimentRunner
from inference.experiments.types import ExperimentConfig


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file, **csv_writer_kwargs()))


def test_matrix_csv_writer_initializes_and_updates_single_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "experiment" / "20260311T143022.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["alias-a", "alias-b"])
    c1 = canonical_prompt_spec("prompt-1")
    c2 = canonical_prompt_spec("prompt-2")
    prompt_id_1 = compute_prompt_id(c1)
    prompt_id_2 = compute_prompt_id(c2)

    writer.initialize(prompts=["prompt-1", "prompt-2"])
    writer.write_cell(
        prompt_id=prompt_id_1,
        alias="alias-b",
        cell=MatrixCell(status=CellStatus.RATE_LIMITED, error_message="retry-after=7200"),
    )

    rows = _read_rows(csv_path)
    pending_cell = json.dumps({"status": "pending"}, separators=(",", ":"))

    assert len(rows) == 2
    assert rows[0]["prompt_id"] == prompt_id_1
    assert rows[0]["prompt"] == serialize_prompt_content(c1)
    assert rows[0]["alias-a"] == pending_cell
    assert rows[0]["alias-b"] == json.dumps(
        {"status": "rate_limited", "error_message": "retry-after=7200"},
        separators=(",", ":"),
    )
    assert rows[1]["prompt_id"] == prompt_id_2
    assert rows[1]["prompt"] == serialize_prompt_content(c2)
    assert rows[1]["alias-a"] == pending_cell
    assert rows[1]["alias-b"] == pending_cell


def test_build_dataframe_from_csv_matches_matrix_contents(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "experiment" / "20260311T143022.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["alias-a", "alias-b"])
    prompt = "prompt-1"
    c = canonical_prompt_spec(prompt)
    prompt_id = compute_prompt_id(c)

    writer.initialize(prompts=[prompt])
    writer.write_cell(
        prompt_id=prompt_id,
        alias="alias-a",
        cell=MatrixCell(status=CellStatus.SUCCESS, response="alias-a:prompt-1"),
    )
    writer.write_cell(
        prompt_id=prompt_id,
        alias="alias-b",
        cell=MatrixCell(status=CellStatus.FAILED, error_message="boom"),
    )

    dataframe = build_dataframe_from_csv(csv_path)

    assert isinstance(dataframe, pd.DataFrame)
    # v2: prompt column holds canonical serialized prompt (messages JSON); cells have status, response, error_message, metadata
    expected_prompt = serialize_prompt_content(c)
    assert dataframe.to_dict(orient="records") == [
        {
            "prompt_id": prompt_id,
            "prompt": expected_prompt,
            "alias-a": {
                "status": "success",
                "response": "alias-a:prompt-1",
                "error_message": None,
                "metadata": None,
            },
            "alias-b": {
                "status": "failed",
                "response": None,
                "error_message": "boom",
                "metadata": None,
            },
        }
    ]


def test_filter_experiment_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "filter_test.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["a", "b", "c"])
    prompts = [
        json.dumps({"system": "S1", "user": "hello"}, separators=(",", ":")),
        json.dumps({"system": "S1", "user": "world"}, separators=(",", ":")),
        json.dumps({"system": "S2", "user": "hello"}, separators=(",", ":")),
    ]
    writer.initialize(prompts=prompts)
    prompt_ids = [compute_prompt_id(canonical_prompt_spec(p)) for p in prompts]
    for i, pid in enumerate(prompt_ids):
        writer.write_cell(pid, "a", MatrixCell(status=CellStatus.SUCCESS, response=f"r-a-{i}"))
        writer.write_cell(pid, "b", MatrixCell(status=CellStatus.SUCCESS, response=f"r-b-{i}"))
        writer.write_cell(
            pid,
            "c",
            MatrixCell(status=CellStatus.SUCCESS, response=f"r-c-{i}")
            if i != 1
            else MatrixCell(status=CellStatus.FAILED, error_message="e"),
        )
    raw = build_dataframe_from_csv(csv_path)
    assert len(raw) == 3

    filtered = filter_experiment_dataframe(raw, models=["a", "b"])
    assert list(filtered.columns) == ["prompt_id", "prompt", "a", "b"]
    assert len(filtered) == 3

    filtered_complete = filter_experiment_dataframe(raw, all_complete=True)
    assert len(filtered_complete) == 2
    assert filtered_complete["prompt_id"].tolist() == [prompt_ids[0], prompt_ids[2]]

    filtered_contains = filter_experiment_dataframe(raw, prompt_contains="hello")
    assert len(filtered_contains) == 2
    filtered_prompt = filter_experiment_dataframe(raw, prompt_contains="S2")
    assert len(filtered_prompt) == 1
    assert filtered_prompt.iloc[0]["prompt_id"] == prompt_ids[2]


def test_to_analysis_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "analysis_test.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["m1", "m2"])
    spec = {"system": "You are helpful.", "user": "What is 2+2?"}
    writer.initialize(prompts=[spec])
    pid = compute_prompt_id(canonical_prompt_spec(spec))
    writer.write_cell(pid, "m1", MatrixCell(status=CellStatus.SUCCESS, response="4"))
    writer.write_cell(pid, "m2", MatrixCell(status=CellStatus.SUCCESS, response="2+2=4"))

    raw = build_dataframe_from_csv(csv_path)
    analysis = to_analysis_dataframe(raw)
    assert list(analysis.columns) == ["prompt_id", "prompt", "m1", "m2"]
    assert "You are helpful." in analysis.iloc[0]["prompt"]
    assert "What is 2+2?" in analysis.iloc[0]["prompt"]
    assert analysis.iloc[0]["m1"] == "4"
    assert analysis.iloc[0]["m2"] == "2+2=4"

    analysis_filtered = to_analysis_dataframe(raw, prompt_contains="2+2")
    assert len(analysis_filtered) == 1
    assert "2+2" in analysis_filtered.iloc[0]["prompt"]


def test_matrix_cell_from_csv_cell_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="invalid status.*Valid:"):
        MatrixCell.from_csv_cell('{"status": "invalid_status"}')


def test_load_existing_state_returns_only_success_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "experiment" / "20260311T143022.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["alias-a", "alias-b"])
    prompt_1 = "prompt-1"
    prompt_2 = "prompt-2"
    prompt_id_1 = compute_prompt_id(canonical_prompt_spec(prompt_1))
    prompt_id_2 = compute_prompt_id(canonical_prompt_spec(prompt_2))

    writer.initialize(prompts=[prompt_1, prompt_2])
    writer.write_cell(
        prompt_id=prompt_id_1,
        alias="alias-a",
        cell=MatrixCell(status=CellStatus.SUCCESS, response="ok-a"),
    )
    writer.write_cell(
        prompt_id=prompt_id_1,
        alias="alias-b",
        cell=MatrixCell(status=CellStatus.FAILED, error_message="boom"),
    )
    writer.write_cell(
        prompt_id=prompt_id_2,
        alias="alias-a",
        cell=MatrixCell(status=CellStatus.RATE_LIMITED, error_message="retry-after=7200"),
    )

    prompt_ids, completed_cells = writer.load_existing_state()

    assert prompt_ids == [prompt_id_1, prompt_id_2]
    assert completed_cells == {prompt_id_1: {"alias-a": CellStatus.SUCCESS}}


class BlockingClient:
    def __init__(self) -> None:
        self.first_completion_written = asyncio.Event()
        self.release_second_cell = asyncio.Event()

    async def complete(self, request: InferenceRequest) -> Any:
        if request.model_alias == "alias-a":
            self.first_completion_written.set()
            return SimpleNamespace(content=f"{request.model_alias}:{request.prompt}")

        await self.release_second_cell.wait()
        return SimpleNamespace(content=f"{request.model_alias}:{request.prompt}")


@pytest.mark.asyncio
async def test_runner_persists_partial_matrix_before_all_cells_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = BlockingClient()
    runner = ExperimentRunner(client=client)  # type: ignore[arg-type]
    config = ExperimentConfig(
        experiment_name="partial-state",
        model_aliases=["alias-a", "alias-b"],
        prompts=["prompt-1"],
    )

    run_task = asyncio.create_task(runner.run(config))
    await asyncio.wait_for(client.first_completion_written.wait(), timeout=1.0)

    csv_files = list((tmp_path / "logs" / "partial-state").glob("*.csv"))
    assert len(csv_files) == 1
    rows = await _wait_for_partial_state(csv_files[0])
    # v2: prompt column + model columns with JSON cells
    assert len(rows) == 1
    assert "prompt_id" in rows[0] and "prompt" in rows[0]
    assert rows[0]["alias-a"] == json.dumps(
        {"status": "success", "response": "alias-a:prompt-1"},
        separators=(",", ":"),
    )
    assert rows[0]["alias-b"] == json.dumps({"status": "pending"}, separators=(",", ":"))

    client.release_second_cell.set()
    result = await run_task

    assert result.csv_path.resolve() == csv_files[0].resolve()


async def _wait_for_partial_state(csv_path: Path) -> list[dict[str, str]]:
    """Wait until the first row has alias-a cell with status success (v2 JSON cells)."""
    for _ in range(100):
        rows = _read_rows(csv_path)
        if rows:
            raw = rows[0].get("alias-a", "")
            try:
                cell = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
                if cell.get("status") == "success":
                    return rows
            except (json.JSONDecodeError, TypeError):
                pass
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for partial CSV persistence")
