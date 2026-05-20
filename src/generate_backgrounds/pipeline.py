"""Three-phase pipeline for generating LLM conversation history backgrounds."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
_logger = logging.getLogger(__name__)


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
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")


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


def _load_name_gender_map(mapping_dir: Path) -> dict[str, str]:
    """Load gender_name.csv → {name: gender} for cross-dimension filtering."""
    import csv as _csv

    csv_path = mapping_dir / "gender_name.csv"
    if not csv_path.exists():
        return {}
    result: dict[str, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            result[row["Name"].strip()] = row["Gender"].strip()
    return result


def _filter_by_gender(
    records: list[BackgroundRecord],
    gender_value: str | None,
    name_to_gender: dict[str, str],
) -> list[BackgroundRecord]:
    """Keep only records whose Name indicator matches the persona gender.

    Records without a Name indicator pass through unfiltered.
    """
    if not gender_value or not name_to_gender:
        return records
    return [
        r for r in records
        if "Name" not in r.indicators
        or name_to_gender.get(r.indicators["Name"]) == gender_value
    ]


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

    def count_pending(self, dimensions: list[str] | None = None) -> dict[str, int]:
        """Return {dimension: pending_combo_count} without making any LLM calls."""
        from generate_backgrounds.combination import load_combinations
        from generate_backgrounds.rendering import (
            discover_dimensions,
            find_dimension_csv,
            load_templates,
        )

        templates = load_templates(self._config.templates_path)
        available = discover_dimensions(self._config.mapping_dir, templates)
        if dimensions is not None:
            requested = {d.lower() for d in dimensions}
            available = [d for d in available if d.lower() in requested]

        result: dict[str, int] = {}
        for dimension in available:
            csv_path = find_dimension_csv(self._config.mapping_dir, dimension)
            assert csv_path is not None
            combos = load_combinations(csv_path, dimension)
            seen_ids = load_existing_ids(self._config.output_dir, dimension)
            result[dimension] = sum(1 for c in combos if c.combination_id not in seen_ids)
        return result

    async def run(
        self,
        dimensions: list[str] | None = None,
        include_partial: bool = False,
    ) -> PipelineResult:
        """Run all three phases: generation, persona enumeration, history assembly."""
        gen_result = await self.run_generation(dimensions)
        assembly = self.assemble_personas(include_partial=include_partial)
        return PipelineResult(dimension_results=gen_result, assembly=assembly)

    async def run_generation(
        self,
        dimensions: list[str] | None = None,
        on_combo_done: Callable[[str, BackgroundRecord | None], None] | None = None,
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

            output_path = self._config.output_dir / dimension / f"{dimension.lower()}.jsonl"
            writer = _JsonlWriter(output_path)
            semaphore = asyncio.Semaphore(self._config.concurrency)

            seen_ids = load_existing_ids(self._config.output_dir, dimension)
            pending = [c for c in combos if c.combination_id not in seen_ids]
            skipped = len(combos) - len(pending)

            total_generated = 0
            max_passes = 3
            for pass_num in range(1, max_passes + 1):
                if not pending:
                    break
                if pass_num > 1:
                    _logger.info("Retry pass %d/%d for %d failed %s combo(s)",
                                 pass_num, max_passes, len(pending), dimension)

                tasks = [
                    asyncio.create_task(
                        self._generate_one(combo, template, semaphore, writer)
                    )
                    for combo in pending
                ]

                failed_combos: list[object] = []
                is_last_pass = pass_num == max_passes
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    try:
                        record = await coro
                    except Exception:
                        record = None
                    if isinstance(record, BackgroundRecord):
                        total_generated += 1
                        if on_combo_done is not None:
                            on_combo_done(dimension, record)
                    else:
                        failed_combos.append(pending[i] if i < len(pending) else None)
                        if is_last_pass and on_combo_done is not None:
                            on_combo_done(dimension, None)

                # Reload seen ids and rebuild pending for next pass
                seen_ids = load_existing_ids(self._config.output_dir, dimension)
                pending = [c for c in combos if c.combination_id not in seen_ids]

            failed = len(combos) - skipped - total_generated

            results.append(
                DimensionResult(
                    dimension=dimension,
                    total=len(combos),
                    generated=total_generated,
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
            _logger.warning("render failed for %s: %s", combo.combination_id, e)
            return None

        async with semaphore:
            if self._config.verbose:
                _logger.debug("starting %s/%s %s…", combo.dimension, combo.dimension_value, combo.combination_id[:12])
            try:
                request = InferenceRequest(
                    model_alias=self._config.model_alias,
                    prompt=rendered,
                    system_prompt=self._config.system_prompt,
                )
                result = await self._client.complete(request)
            except Exception as e:
                root = e.__cause__ or e
                _logger.warning("LLM call failed for %s: %s: %s", combo.combination_id, type(root).__name__, root)
                return None

        if self._config.verbose:
            _logger.debug("done %s/%s %s…", combo.dimension, combo.dimension_value, combo.combination_id[:12])

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

    def assemble_personas(
        self,
        on_history_done: Callable[[bool], None] | None = None,
        on_total: Callable[[int], None] | None = None,
        persona_filter: dict[str, str] | None = None,
        include_partial: bool = False,
    ) -> AssemblyResult:
        """Phase 2 + 3: enumerate personas and assemble conversation histories.

        Args:
            persona_filter: If given, only assemble histories for personas that
                match **all** specified dimension=value pairs.  Dimensions not
                mentioned in the filter are unconstrained (any value or None).
                Example: ``{"Gender": "Male", "Age": "Young"}``
            include_partial: If False (default), only assemble histories for
                personas where every dimension has a value (no Nones).
                If True, also include partial personas.
        """
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

        # Pre-group records by (dimension, dimension_value) for O(1) lookups
        grouped: dict[str, dict[str, list[BackgroundRecord]]] = {}
        for dim, records in dim_backgrounds.items():
            by_value: dict[str, list[BackgroundRecord]] = {}
            for r in records:
                by_value.setdefault(r.dimension_value, []).append(r)
            grouped[dim] = by_value

        # Phase 2: enumerate personas = Cartesian product of dimension_values,
        # with None as an option for each dimension (= "exclude this dimension").
        # The all-None persona (no dimensions at all) is filtered out.
        dim_order = list(dim_backgrounds.keys())
        dim_value_sets_with_none: list[list[str | None]] = [
            [None] + list(dict.fromkeys(r.dimension_value for r in dim_backgrounds[d]))
            for d in dim_order
        ]
        if include_partial:
            personas = [
                p for p in itertools.product(*dim_value_sets_with_none)
                if any(v is not None for v in p)
            ]
        else:
            personas = [
                p for p in itertools.product(*dim_value_sets_with_none)
                if all(v is not None for v in p)
            ]

        # Apply persona filter if provided
        if persona_filter:
            personas = [
                p for p in personas
                if all(
                    dict(zip(dim_order, p)).get(dim) == val
                    for dim, val in persona_filter.items()
                )
            ]

        total_personas = len(personas)

        # Load cross-dimension gender→name constraint
        name_to_gender = _load_name_gender_map(self._config.mapping_dir)

        # Pre-count total histories for progress reporting
        _total_expected = 0
        for persona_tuple in personas:
            persona_tmp: dict[str, str | None] = dict(zip(dim_order, persona_tuple))
            included = [d for d in dim_order if persona_tmp[d] is not None]
            gender_val = persona_tmp.get("Gender")
            counts = [
                len(_filter_by_gender(grouped[d][persona_tmp[d]], gender_val, name_to_gender)
                    if d != "Gender" else grouped[d][persona_tmp[d]])
                for d in included
            ]
            product = 1
            for c in counts:
                product *= c
            _total_expected += product

        if on_total is not None:
            on_total(_total_expected)

        # Phase 3: for each persona, cross indicator combos across included dimensions
        seen_history_ids = load_existing_history_ids(self._config.personas_dir)
        output_path = self._config.personas_dir / "personas.jsonl"
        writer = _JsonlWriter(output_path)

        total_histories = 0
        generated_histories = 0
        skipped_histories = 0

        for persona_tuple in personas:
            # Full persona dict with None for excluded dimensions
            persona: dict[str, str | None] = dict(zip(dim_order, persona_tuple))

            # Only collect records for included (non-None) dimensions
            included_dims = [d for d in dim_order if persona[d] is not None]
            gender_value = persona.get("Gender")
            per_dim_records: list[list[BackgroundRecord]] = [
                _filter_by_gender(grouped[d][persona[d]], gender_value, name_to_gender)
                if d != "Gender" else grouped[d][persona[d]]
                for d in included_dims
            ]

            for combo_tuple in itertools.product(*per_dim_records):
                total_histories += 1
                combination_ids = {r.dimension: r.combination_id for r in combo_tuple}
                hid = _history_id(combination_ids)

                if hid in seen_history_ids:
                    skipped_histories += 1
                    if on_history_done is not None:
                        on_history_done(False)
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
                if on_history_done is not None:
                    on_history_done(True)

        return AssemblyResult(
            total_personas=total_personas,
            total_histories=_total_expected,
            generated_histories=generated_histories,
            skipped_histories=skipped_histories,
        )
