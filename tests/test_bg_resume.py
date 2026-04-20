"""Tests for background pipeline resume/deduplication behaviour."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from generate_backgrounds.pipeline import BackgroundPipeline, GenerationConfig
from inference.client import InferenceRequest


class CountingClient:
    """Tracks how many LLM calls are made."""

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, request: InferenceRequest) -> SimpleNamespace:
        self.call_count += 1
        return SimpleNamespace(content=f"response:{self.call_count}")


def _write_csv(path: Path, dimension: str, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dimension_value", "Indicator_name", "Indicator_value"])
        writer.writerows(rows)


def _write_templates(path: Path, templates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(templates), encoding="utf-8")


def _make_pipeline(tmp_path: Path, client: CountingClient) -> BackgroundPipeline:
    mapping_dir = tmp_path / "mapping"
    _write_csv(
        mapping_dir / "dima.csv",
        "DimA",
        [
            ("val1", "Trait", "alpha"),
            ("val2", "Trait", "beta"),
        ],
    )
    templates_path = tmp_path / "templates.json"
    _write_templates(templates_path, {"DimA": "My trait is <Trait>."})

    config = GenerationConfig(
        model_alias="mock-test",
        templates_path=templates_path,
        mapping_dir=mapping_dir,
        output_dir=tmp_path / "backgrounds",
        personas_dir=tmp_path / "personas",
        concurrency=4,
    )
    return BackgroundPipeline(client=client, config=config)


@pytest.mark.asyncio
async def test_second_run_skips_generation(tmp_path: Path) -> None:
    """Running the pipeline twice should not regenerate backgrounds."""
    client = CountingClient()
    pipeline = _make_pipeline(tmp_path, client)

    # First run: generates backgrounds
    result1 = await pipeline.run()
    calls_first_run = client.call_count
    assert calls_first_run > 0
    assert result1.dimension_results[0].generated > 0

    # Second run: same pipeline, should skip all backgrounds
    pipeline2 = _make_pipeline(tmp_path, client)
    result2 = await pipeline2.run()
    calls_second_run = client.call_count - calls_first_run

    assert calls_second_run == 0
    assert result2.dimension_results[0].skipped == result1.dimension_results[0].total
    assert result2.dimension_results[0].generated == 0


@pytest.mark.asyncio
async def test_second_run_skips_persona_assembly(tmp_path: Path) -> None:
    """Running the pipeline twice should not re-assemble existing personas."""
    client = CountingClient()
    pipeline = _make_pipeline(tmp_path, client)

    result1 = await pipeline.run()
    assert result1.assembly.generated_histories > 0

    # Second run
    pipeline2 = _make_pipeline(tmp_path, client)
    result2 = await pipeline2.run()

    assert result2.assembly.skipped_histories == result1.assembly.total_histories
    assert result2.assembly.generated_histories == 0


@pytest.mark.asyncio
async def test_partial_resume_generates_only_missing(tmp_path: Path) -> None:
    """If one background exists and another is added, only the new one is generated."""
    client = CountingClient()

    # First run with one dimension value
    mapping_dir = tmp_path / "mapping"
    _write_csv(mapping_dir / "dima.csv", "DimA", [("val1", "Trait", "alpha")])
    templates_path = tmp_path / "templates.json"
    _write_templates(templates_path, {"DimA": "My trait is <Trait>."})

    config = GenerationConfig(
        model_alias="mock-test",
        templates_path=templates_path,
        mapping_dir=mapping_dir,
        output_dir=tmp_path / "backgrounds",
        personas_dir=tmp_path / "personas",
        concurrency=4,
    )
    pipeline = BackgroundPipeline(client=client, config=config)
    await pipeline.run()
    calls_after_first = client.call_count
    assert calls_after_first == 1  # one combo = one LLM call

    # Now add a second dimension value and re-run
    _write_csv(
        mapping_dir / "dima.csv",
        "DimA",
        [
            ("val1", "Trait", "alpha"),
            ("val2", "Trait", "beta"),
        ],
    )
    pipeline2 = BackgroundPipeline(client=client, config=config)
    result = await pipeline2.run()

    calls_second_run = client.call_count - calls_after_first
    assert calls_second_run == 1  # only the new combo
    assert result.dimension_results[0].skipped == 1
    assert result.dimension_results[0].generated == 1


@pytest.mark.asyncio
async def test_corrupted_line_in_output_does_not_crash(tmp_path: Path) -> None:
    """Malformed JSON lines in existing output are gracefully skipped."""
    client = CountingClient()
    pipeline = _make_pipeline(tmp_path, client)

    # First run to create output
    await pipeline.run()

    # Append a corrupted line to the output
    dim_dir = tmp_path / "backgrounds" / "DimA"
    jsonl_files = list(dim_dir.glob("*.jsonl"))
    assert len(jsonl_files) > 0
    with open(jsonl_files[0], "a", encoding="utf-8") as f:
        f.write("{corrupted json\n")

    # Second run should not crash and still skip already-generated combos
    pipeline2 = _make_pipeline(tmp_path, client)
    result = await pipeline2.run()

    assert result.dimension_results[0].generated == 0
