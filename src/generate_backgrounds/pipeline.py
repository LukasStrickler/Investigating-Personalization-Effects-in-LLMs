"""Three-phase pipeline for generating LLM conversation history backgrounds."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent


@dataclass(frozen=True)
class GenerationConfig:
    model_alias: str
    templates_path: Path = field(
        default_factory=lambda: _HERE / "dimension_templates.json"
    )
    mapping_dir: Path = field(
        default_factory=lambda: _HERE / "dimension_value_mapping"
    )
    output_dir: Path = field(
        default_factory=lambda: _HERE / "data" / "backgrounds"
    )
    personas_dir: Path = field(
        default_factory=lambda: _HERE / "data" / "personas"
    )
    concurrency: int = 8
    system_prompt: str | None = None


@dataclass(frozen=True)
class BackgroundRecord:
    schema_version: int
    dimension: str
    dimension_value: str
    combination_id: str
    indicators: dict[str, str]
    model_alias: str
    messages: list[dict[str, str]]
    generated_at: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dimension": self.dimension,
            "dimension_value": self.dimension_value,
            "combination_id": self.combination_id,
            "indicators": self.indicators,
            "model_alias": self.model_alias,
            "messages": self.messages,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class ConversationHistory:
    history_id: str
    persona: dict[str, str | None]  # ALL known dimensions; None = excluded from this history
    combination_ids: dict[str, str]
    messages: list[dict[str, str]]
    generated_at: str

    def to_dict(self) -> dict:
        return {
            "history_id": self.history_id,
            "persona": self.persona,
            "combination_ids": self.combination_ids,
            "messages": self.messages,
            "generated_at": self.generated_at,
        }


@dataclass
class DimensionResult:
    dimension: str
    total: int
    generated: int
    skipped: int
    failed: int


@dataclass
class AssemblyResult:
    total_personas: int
    total_histories: int
    generated_histories: int
    skipped_histories: int


@dataclass
class PipelineResult:
    dimension_results: list[DimensionResult]
    assembly: AssemblyResult


def _history_id(combination_ids: dict[str, str]) -> str:
    payload = json.dumps(combination_ids, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ts_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".jsonl"


def load_existing_ids(output_dir: Path, dimension: str) -> set[str]:
    """Collect all combination_ids already persisted under output_dir/dimension/."""
    dim_dir = output_dir / dimension
    seen: set[str] = set()
    if not dim_dir.exists():
        return seen
    for jsonl_file in dim_dir.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("combination_id")
                    if cid:
                        seen.add(cid)
                except json.JSONDecodeError:
                    pass
    return seen


def load_existing_history_ids(personas_dir: Path) -> set[str]:
    """Collect all history_ids already persisted under personas_dir."""
    seen: set[str] = set()
    if not personas_dir.exists():
        return seen
    for jsonl_file in personas_dir.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    hid = obj.get("history_id")
                    if hid:
                        seen.add(hid)
                except json.JSONDecodeError:
                    pass
    return seen


def load_all_backgrounds(output_dir: Path, dimension: str) -> list[BackgroundRecord]:
    """Load all persisted BackgroundRecords for a dimension."""
    dim_dir = output_dir / dimension
    records: list[BackgroundRecord] = []
    if not dim_dir.exists():
        return records
    for jsonl_file in sorted(dim_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(
                        BackgroundRecord(
                            schema_version=obj.get("schema_version", 1),
                            dimension=obj["dimension"],
                            dimension_value=obj["dimension_value"],
                            combination_id=obj["combination_id"],
                            indicators=obj["indicators"],
                            model_alias=obj["model_alias"],
                            messages=obj["messages"],
                            generated_at=obj["generated_at"],
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    pass
    return records


class _JsonlWriter:
    """Thread-safe append writer for a JSONL file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()

    def append(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)


class BackgroundPipeline:
    def __init__(
        self,
        *,
        client,  # UnifiedInferenceClient — typed loosely to avoid circular import
        config: GenerationConfig,
    ) -> None:
        self._client = client
        self._config = config

    async def run(self, dimensions: list[str] | None = None) -> PipelineResult:
        """Run all three phases: generation, persona enumeration, history assembly."""
        gen_result = await self.run_generation(dimensions)
        assembly = self.assemble_personas()
        return PipelineResult(dimension_results=gen_result, assembly=assembly)

    async def run_generation(
        self, dimensions: list[str] | None = None
    ) -> list[DimensionResult]:
        """Phase 1: generate per-dimension background records via LLM calls."""
        from generate_backgrounds.combination import load_combinations
        from generate_backgrounds.rendering import (
            discover_dimensions,
            find_dimension_csv,
            load_templates,
        )

        templates = load_templates(self._config.templates_path)
        available = discover_dimensions(self._config.mapping_dir, templates)

        if dimensions is not None:
            # Normalize to lowercase for comparison
            requested = {d.lower() for d in dimensions}
            selected = [d for d in available if d.lower() in requested]
        else:
            selected = available

        results: list[DimensionResult] = []
        for dimension in selected:
            csv_path = find_dimension_csv(self._config.mapping_dir, dimension)
            assert csv_path is not None
            combos = load_combinations(csv_path, dimension)
            template = templates[dimension]

            seen_ids = load_existing_ids(self._config.output_dir, dimension)
            pending = [c for c in combos if c.combination_id not in seen_ids]
            skipped = len(combos) - len(pending)

            output_path = self._config.output_dir / dimension / _ts_filename()
            writer = _JsonlWriter(output_path)

            semaphore = asyncio.Semaphore(self._config.concurrency)
            tasks = [
                asyncio.create_task(
                    self._generate_one(combo, template, semaphore, writer)
                )
                for combo in pending
            ]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            generated = sum(1 for r in task_results if isinstance(r, BackgroundRecord))
            failed = len(task_results) - generated

            results.append(
                DimensionResult(
                    dimension=dimension,
                    total=len(combos),
                    generated=generated,
                    skipped=skipped,
                    failed=failed,
                )
            )

        return results

    async def _generate_one(
        self,
        combo,
        template: str,
        semaphore: asyncio.Semaphore,
        writer: _JsonlWriter,
    ) -> BackgroundRecord | None:
        from generate_backgrounds.rendering import render_template
        from inference import InferenceRequest

        try:
            rendered = render_template(template, combo.indicators)
        except Exception as e:
            print(f"[warn] render failed for {combo.combination_id}: {e}")
            return None

        async with semaphore:
            try:
                request = InferenceRequest(
                    model_alias=self._config.model_alias,
                    prompt=rendered,
                    system_prompt=self._config.system_prompt,
                )
                result = await self._client.complete(request)
            except Exception as e:
                print(f"[warn] LLM call failed for {combo.combination_id}: {e}")
                return None

        record = BackgroundRecord(
            schema_version=1,
            dimension=combo.dimension,
            dimension_value=combo.dimension_value,
            combination_id=combo.combination_id,
            indicators=combo.indicators,
            model_alias=self._config.model_alias,
            messages=[
                {"role": "user", "content": rendered},
                {"role": "assistant", "content": result.content},
            ],
            generated_at=_now_iso(),
        )
        await asyncio.to_thread(writer.append, record.to_dict())
        return record

    def assemble_personas(self) -> AssemblyResult:
        """Phase 2 + 3: enumerate personas and assemble conversation histories."""
        from generate_backgrounds.rendering import discover_dimensions, load_templates

        templates = load_templates(self._config.templates_path)
        available = discover_dimensions(self._config.mapping_dir, templates)

        # Load all backgrounds grouped by dimension
        dim_backgrounds: dict[str, list[BackgroundRecord]] = {}
        for dimension in available:
            records = load_all_backgrounds(self._config.output_dir, dimension)
            if records:
                dim_backgrounds[dimension] = records

        if not dim_backgrounds:
            return AssemblyResult(
                total_personas=0,
                total_histories=0,
                generated_histories=0,
                skipped_histories=0,
            )

        # Phase 2: enumerate personas = Cartesian product of dimension_values,
        # with None as an option for each dimension (= "exclude this dimension").
        # The all-None persona (no dimensions at all) is filtered out.
        dim_order = list(dim_backgrounds.keys())
        dim_value_sets_with_none: list[list[str | None]] = [
            [None] + list(dict.fromkeys(r.dimension_value for r in dim_backgrounds[d]))
            for d in dim_order
        ]
        personas = [
            p for p in itertools.product(*dim_value_sets_with_none)
            if any(v is not None for v in p)
        ]
        total_personas = len(personas)

        # Phase 3: for each persona, cross indicator combos across included dimensions
        seen_history_ids = load_existing_history_ids(self._config.personas_dir)
        output_path = self._config.personas_dir / _ts_filename()
        writer = _JsonlWriter(output_path)

        total_histories = 0
        generated_histories = 0
        skipped_histories = 0

        for persona_tuple in personas:
            # Full persona dict with None for excluded dimensions
            persona: dict[str, str | None] = dict(zip(dim_order, persona_tuple))

            # Only collect records for included (non-None) dimensions
            included_dims = [d for d in dim_order if persona[d] is not None]
            per_dim_records: list[list[BackgroundRecord]] = [
                [r for r in dim_backgrounds[d] if r.dimension_value == persona[d]]
                for d in included_dims
            ]

            for combo_tuple in itertools.product(*per_dim_records):
                total_histories += 1
                combination_ids = {r.dimension: r.combination_id for r in combo_tuple}
                hid = _history_id(combination_ids)

                if hid in seen_history_ids:
                    skipped_histories += 1
                    continue

                messages: list[dict[str, str]] = []
                for record in combo_tuple:
                    messages.extend(record.messages)

                history = ConversationHistory(
                    history_id=hid,
                    persona=persona,
                    combination_ids=combination_ids,
                    messages=messages,
                    generated_at=_now_iso(),
                )
                writer.append(history.to_dict())
                seen_history_ids.add(hid)
                generated_histories += 1

        return AssemblyResult(
            total_personas=total_personas,
            total_histories=total_histories,
            generated_histories=generated_histories,
            skipped_histories=skipped_histories,
        )
