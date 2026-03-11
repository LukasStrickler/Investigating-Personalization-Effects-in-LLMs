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
    compute_prompt_id,
    csv_writer_kwargs,
)
from inference.experiments.dataframe import build_dataframe_from_csv
from inference.experiments.persistence import MatrixCSVWriter
from inference.experiments.runner import ExperimentRunner
from inference.experiments.types import ExperimentConfig


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file, **csv_writer_kwargs()))


def test_matrix_csv_writer_initializes_and_updates_single_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "experiment" / "20260311T143022.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["alias-a", "alias-b"])
    prompt_id = compute_prompt_id("prompt-1")

    writer.initialize(prompts=["prompt-1", "prompt-2"])
    writer.write_cell(
        prompt_id=prompt_id,
        alias="alias-b",
        cell=MatrixCell(status=CellStatus.RATE_LIMITED, error_message="retry-after=7200"),
    )

    rows = _read_rows(csv_path)

    assert rows == [
        {
            "prompt_id": prompt_id,
            "alias-a": "",
            "alias-b": json.dumps(
                {"status": "rate_limited", "error_message": "retry-after=7200"},
                separators=(",", ":"),
            ),
        },
        {"prompt_id": compute_prompt_id("prompt-2"), "alias-a": "", "alias-b": ""},
    ]


def test_build_dataframe_from_csv_matches_matrix_contents(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "experiment" / "20260311T143022.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["alias-a", "alias-b"])
    prompt = "prompt-1"
    prompt_id = compute_prompt_id(prompt)

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
    assert dataframe.to_dict(orient="records") == [
        {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "alias-a": {
                "status": "success",
                "response": "alias-a:prompt-1",
                "error_message": None,
            },
            "alias-b": {"status": "failed", "response": None, "error_message": "boom"},
        }
    ]


def test_load_existing_state_returns_only_success_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "experiment" / "20260311T143022.csv"
    writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=["alias-a", "alias-b"])
    prompt_1 = "prompt-1"
    prompt_2 = "prompt-2"
    prompt_id_1 = compute_prompt_id(prompt_1)
    prompt_id_2 = compute_prompt_id(prompt_2)

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
    assert rows == [
        {
            "prompt_id": rows[0]["prompt_id"],
            "alias-a": json.dumps(
                {"status": "success", "response": "alias-a:prompt-1"},
                separators=(",", ":"),
            ),
            "alias-b": "",
        }
    ]

    client.release_second_cell.set()
    result = await run_task

    assert result.csv_path.resolve() == csv_files[0].resolve()


async def _wait_for_partial_state(csv_path: Path) -> list[dict[str, str]]:
    for _ in range(100):
        rows = _read_rows(csv_path)
        if rows and rows[0]["alias-a"]:
            return rows
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for partial CSV persistence")
