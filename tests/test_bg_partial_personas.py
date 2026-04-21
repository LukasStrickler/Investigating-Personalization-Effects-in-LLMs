"""Tests for partial-combination persona assembly (None = excluded dimension)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from generate_backgrounds.pipeline import BackgroundPipeline, GenerationConfig
from inference.client import InferenceRequest


class FakeClient:
    async def complete(self, request: InferenceRequest) -> SimpleNamespace:
        return SimpleNamespace(content=f"mock:{request.prompt[:30]}")


def _write_csv(path: Path, dimension: str, rows: list[tuple[str, str, str]]) -> None:
    """Write a minimal dimension_value_mapping CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dimension_value", "Indicator_name", "Indicator_value"])
        writer.writerows(rows)


def _write_templates(path: Path, templates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(templates), encoding="utf-8")


@pytest.fixture()
def two_dim_pipeline(tmp_path: Path) -> BackgroundPipeline:
    """Pipeline with two tiny dimensions (DimA: val1/val2, DimB: valX/valY)."""
    mapping_dir = tmp_path / "mapping"
    _write_csv(
        mapping_dir / "dima.csv",
        "DimA",
        [
            ("val1", "Trait", "alpha"),
            ("val2", "Trait", "beta"),
        ],
    )
    _write_csv(
        mapping_dir / "dimb.csv",
        "DimB",
        [
            ("valX", "Feature", "one"),
            ("valY", "Feature", "two"),
        ],
    )

    templates_path = tmp_path / "templates.json"
    _write_templates(
        templates_path,
        {
            "DimA": "My trait is <Trait>.",
            "DimB": "My feature is <Feature>.",
        },
    )

    config = GenerationConfig(
        model_alias="mock-test",
        templates_path=templates_path,
        mapping_dir=mapping_dir,
        output_dir=tmp_path / "backgrounds",
        personas_dir=tmp_path / "personas",
        concurrency=4,
    )
    return BackgroundPipeline(client=FakeClient(), config=config)


@pytest.mark.asyncio
async def test_full_run_produces_partial_combinations(
    two_dim_pipeline: BackgroundPipeline,
) -> None:
    result = await two_dim_pipeline.run()

    # 2 dimensions × 2 values each → (2+1)*(2+1) - 1 = 8 personas
    assert result.assembly.total_personas == 8

    personas_dir: Path = two_dim_pipeline._config.personas_dir
    histories = []
    for jsonl_file in sorted(personas_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    histories.append(json.loads(line))

    assert len(histories) == result.assembly.total_histories

    persona_keys = {tuple(sorted(h["persona"].items())) for h in histories}

    # Every history must list both dimensions in persona
    for h in histories:
        assert set(h["persona"].keys()) == {"DimA", "DimB"}, (
            f"persona is missing a dimension: {h['persona']}"
        )

    # Full combinations (both non-null) must exist
    assert any(
        h["persona"]["DimA"] is not None and h["persona"]["DimB"] is not None
        for h in histories
    ), "Expected at least one full (both dims non-null) history"

    # Partial: DimA active, DimB excluded
    assert any(
        h["persona"]["DimA"] is not None and h["persona"]["DimB"] is None
        for h in histories
    ), "Expected histories where DimA is set and DimB is null"

    # Partial: DimB active, DimA excluded
    assert any(
        h["persona"]["DimA"] is None and h["persona"]["DimB"] is not None
        for h in histories
    ), "Expected histories where DimB is set and DimA is null"

    # No all-null persona
    assert not any(
        h["persona"]["DimA"] is None and h["persona"]["DimB"] is None
        for h in histories
    ), "All-null persona should be excluded"

    # combination_ids must only reference included (non-null) dimensions
    for h in histories:
        active_dims = {d for d, v in h["persona"].items() if v is not None}
        assert set(h["combination_ids"].keys()) == active_dims, (
            f"combination_ids keys don't match active dims: {h}"
        )


@pytest.mark.asyncio
async def test_single_dim_unchanged(tmp_path: Path) -> None:
    """With one dimension, all personas must have that dim set (no regression)."""
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
    )
    pipeline = BackgroundPipeline(client=FakeClient(), config=config)
    result = await pipeline.run()

    # (2+1) - 1 = 2 personas (excluding all-None)
    assert result.assembly.total_personas == 2

    histories = []
    for jsonl_file in sorted((tmp_path / "personas").glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    histories.append(json.loads(line))

    assert len(histories) == 2
    for h in histories:
        assert h["persona"]["DimA"] is not None
