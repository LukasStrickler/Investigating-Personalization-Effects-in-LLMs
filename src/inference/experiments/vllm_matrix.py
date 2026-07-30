"""Cluster glue for run_cluster_direct_probing_stage1.py (probe, circuit breaker, personas)."""

from __future__ import annotations

import asyncio
import json
import random
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.client import UnifiedInferenceClient
from inference.config import load_config_from_file, resolve_provider_base_url
from inference.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    PromptSpec,
)
from inference.experiments.csv_schema import canonical_prompt_spec, compute_prompt_id
from inference.experiments.persistence import MatrixCSVWriter, load_existing_matrix
from inference.judges.types import JudgeSubject
from inference.providers import (
    LiteLLMProviderAdapter,
    ProviderAdapter,
    ProviderRequest,
    ProviderResponse,
)

DIRECT_PROBING_STAGE1_PROBE = (
    "Based on our conversation, what is my gender and cultural background?"
)
DIRECT_PROBING_JUDGE_PROMPT = (
    "Based on this response, determine the gender and cultural background/region "
    "of the speaker. Pick exactly one option."
)

DEMO_PROMPTS: list[PromptSpec] = [
    {
        "messages": [
            {"role": "user", "content": "In one sentence, what is a large language model?"}
        ],
        "metadata": {"demo_id": "d1"},
    },
    {
        "messages": [{"role": "user", "content": "Name a primary color."}],
        "metadata": {"demo_id": "d2"},
    },
    {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "metadata": {"demo_id": "d3"},
    },
]


class CircuitTrippedError(RuntimeError):
    """Raised when the connection circuit breaker has tripped."""


def is_connection_error(exc: BaseException) -> bool:
    """True if any error in the cause/context chain looks like a transport failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        if "connect" in name or "timeout" in name or "connection" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


class ConnectionCircuitBreaker:
    """Fail fast after consecutive transport errors (vLLM server outage mid-run)."""

    def __init__(self, inner: ProviderAdapter, *, threshold: int = 5) -> None:
        self._inner = inner
        self._threshold = threshold
        self._consecutive = 0
        self._tripped = False
        self._lock = asyncio.Lock()

    @property
    def tripped(self) -> bool:
        return self._tripped

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self._tripped:
            raise CircuitTrippedError(
                f"circuit breaker tripped after {self._threshold} consecutive connection "
                "errors — the server appears to be down; aborting the phase"
            )
        try:
            response = await self._inner.complete(request)
        except Exception as error:
            async with self._lock:
                if is_connection_error(error):
                    self._consecutive += 1
                    if self._consecutive >= self._threshold:
                        self._tripped = True
                else:
                    self._consecutive = 0
            raise
        async with self._lock:
            self._consecutive = 0
        return response


def probe_openai_compatible_server(
    base_url: str,
    *,
    served_model: str | None = None,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Check that an OpenAI-compatible server is reachable at ``base_url``."""
    models_url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(models_url, headers={"Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, ValueError) as error:
        return False, f"cannot reach a server at {models_url}: {error}"
    try:
        served = [entry.get("id") for entry in json.loads(body).get("data", [])]
    except (ValueError, AttributeError) as error:
        return False, f"server at {models_url} returned unparseable /models: {error}"
    if served_model is not None and served_model not in served:
        return False, f"server is up but does not serve {served_model!r}; it serves {served}"
    return True, f"server reachable at {models_url}; serves {served}"


def _repo_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "config" / "inference.example.yaml").exists():
            return candidate
    return Path.cwd()


def default_personas_path() -> Path:
    return _repo_root() / "src" / "generate_backgrounds" / "data" / "personas" / "personas.jsonl"


def load_direct_probing_stage1(
    *,
    sample_per_group: int = 10_000,
    seed: int = 123,
    personas_path: Path | None = None,
) -> tuple[list[PromptSpec], list[str]]:
    """Persona probes aligned with ``experiments/run_direct_probing.py`` (seed=123)."""
    path = personas_path or default_personas_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"personas file not found: {path} — clone the full repo and run from repo root"
        )
    all_personas: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            all_personas.append(json.loads(line))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for persona in all_personas:
        gender = persona["persona"].get("Gender")
        region = persona["persona"].get("Region")
        if gender and region:
            grouped[(gender, region)].append(persona)

    all_regions = sorted({region for (_, region) in grouped})
    random.seed(seed)
    sampled: list[dict] = []
    for region in all_regions:
        for gender in ["Male", "Female"]:
            pool = list(grouped[(gender, region)])
            if len(pool) < sample_per_group:
                print(
                    f"WARNING: only {len(pool)} personas for ({gender}, {region}), using all",
                    file=sys.stderr,
                )
                sampled.extend(pool)
            else:
                sampled.extend(random.sample(pool, sample_per_group))

    gender_options = ["Male", "Female"]
    combined_classes = [f"{g} - {r}" for g in gender_options for r in all_regions]
    prompts: list[PromptSpec] = [
        {
            "messages": list(persona["messages"])
            + [{"role": "user", "content": DIRECT_PROBING_STAGE1_PROBE}],
            "metadata": {
                "history_id": persona["history_id"],
                "true_gender": persona["persona"]["Gender"],
                "true_region": persona["persona"]["Region"],
            },
        }
        for persona in sampled
    ]
    return prompts, combined_classes


def build_direct_probing_judge_subjects(
    df: Any,
    *,
    model_alias: str,
    csv_path: Path,
) -> tuple[list[JudgeSubject], int]:
    """Build Stage-2 judge subjects from a Stage-1 matrix CSV column."""
    subjects: list[JudgeSubject] = []
    skipped = 0
    for _, row in df.iterrows():
        meta = row.get("prompt_metadata")
        if not isinstance(meta, dict) or "history_id" not in meta:
            skipped += 1
            continue
        if row[model_alias] is None:
            continue
        subjects.append(
            JudgeSubject(
                subject_id=f"probe-{meta['history_id']}",
                subject_content=str(row[model_alias]),
                subject_model_alias=model_alias,
                source_id=str(csv_path),
                prompt_id=str(row["prompt_id"]),
                metadata=dict(meta),
            )
        )
    return subjects, skipped


def load_matrix_prompts(
    source: str,
    *,
    sample_per_group: int = 10_000,
    seed: int = 123,
) -> list[PromptSpec]:
    """Return prompt specs for a matrix column run."""
    if source == "demo":
        return list(DEMO_PROMPTS)
    if source == "direct-probing":
        prompts, _classes = load_direct_probing_stage1(sample_per_group=sample_per_group, seed=seed)
        return prompts
    raise ValueError(f"unknown prompts source {source!r} (choose: demo, direct-probing)")


def validate_launch_config(config_path: Path, alias: str, served: str) -> None:
    """Verify alias/SERVED against config before starting a vLLM server."""
    config = load_config_from_file(config_path)
    alias_cfg = config.model_aliases.get(alias)
    if alias_cfg is None:
        raise ValueError(f"alias {alias!r} not in {config_path}")
    if alias_cfg.model != served:
        raise ValueError(f"SERVED={served!r} != config alias.model={alias_cfg.model!r}")
    provider = config.providers.get(alias_cfg.provider)
    if provider is None:
        raise ValueError(f"provider {alias_cfg.provider!r} missing from config")
    if provider.name == "vllm" and not provider.base_url:
        raise ValueError("vllm provider has no base_url in config")
    if alias_cfg.provider != "mock" and provider.name != "vllm":
        raise ValueError(
            f"alias {alias!r} uses provider {alias_cfg.provider!r} ({provider.name}); "
            "cluster Stage 1 expects a vllm alias (or mock-test for smoke)"
        )


@dataclass(frozen=True, slots=True)
class MatrixColumnRun:
    """Inputs for one resumable matrix-column execution."""

    config_path: Path
    model_alias: str
    columns: list[str]
    experiment_name: str
    csv_path: Path
    prompts_source: str
    limit: int | None
    sample_per_group: int
    circuit_breaker_threshold: int
    skip_probe: bool


async def run_matrix_column(run: MatrixColumnRun) -> int:
    """Run one matrix column. Returns a process exit code (0 == clean)."""
    if not run.config_path.exists():
        return 2

    config = load_config_from_file(run.config_path)
    alias_cfg = config.model_aliases.get(run.model_alias)
    if alias_cfg is None:
        return 2

    columns = list(
        dict.fromkeys(
            run.columns if run.model_alias in run.columns else [run.model_alias, *run.columns]
        )
    )

    try:
        prompts = load_matrix_prompts(run.prompts_source, sample_per_group=run.sample_per_group)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if run.limit is not None:
        prompts = prompts[: run.limit]
    if not prompts:
        return 4

    provider_cfg = config.providers.get(alias_cfg.provider)
    base_url = resolve_provider_base_url(provider_cfg) if provider_cfg else None
    if alias_cfg.provider == "vllm" and not run.skip_probe and base_url:
        ok, message = probe_openai_compatible_server(base_url, served_model=alias_cfg.model)
        print(f"[probe] {message}")
        if not ok:
            return 3

    run.csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_existed = run.csv_path.exists()
    if not csv_existed:
        MatrixCSVWriter(run.csv_path, columns).initialize(prompts)
        print(f"[init] created matrix CSV at {run.csv_path} with columns {columns}")

    prompt_ids = [compute_prompt_id(canonical_prompt_spec(prompt)) for prompt in prompts]
    run_cells = {(prompt_id, run.model_alias) for prompt_id in prompt_ids}
    seen, completed = load_existing_matrix(run.csv_path)

    current_ids = set(prompt_ids)
    if csv_existed and seen and not seen.issubset(current_ids):
        return 6

    already_success = sum(1 for pid in prompt_ids if (pid, run.model_alias) in completed)
    print(
        f"[plan] column={run.model_alias} prompts={len(prompt_ids)} "
        f"already_success={already_success} to_dispatch={len(prompt_ids) - already_success}"
    )

    marker_path = Path(f"{run.csv_path}.{run.model_alias}.complete")
    if len(prompt_ids) - already_success > 0:
        marker_path.unlink(missing_ok=True)

    experiment_config = ExperimentConfig(
        experiment_name=run.experiment_name,
        model_aliases=columns,
        prompts=prompts,
        run_cells=run_cells,
        resume_from_existing_csv=True,
        existing_csv_path=run.csv_path,
        verbosity="normal",
    )

    breaker = ConnectionCircuitBreaker(
        LiteLLMProviderAdapter(), threshold=run.circuit_breaker_threshold
    )
    client = UnifiedInferenceClient(config=config, adapter=breaker)

    try:
        result = await ExperimentRunner(client).run(experiment_config)
    except CircuitTrippedError:
        return 5

    summary = result.summary
    print(
        f"[done] {result.csv_path} | cells={summary.total_cells} "
        f"completed={summary.completed_cells} failed={summary.failed_cells} "
        f"rate_limited={summary.rate_limited_cells}"
    )

    if breaker.tripped:
        return 5

    _seen_after, completed_after = load_existing_matrix(run.csv_path)
    remaining = [pid for pid in prompt_ids if (pid, run.model_alias) not in completed_after]
    if remaining:
        marker_path.unlink(missing_ok=True)
        return 10

    marker_path.write_text("complete\n", encoding="utf-8")
    print(f"[complete] all {len(prompt_ids)} cells for column {run.model_alias!r} are SUCCESS.")
    return 0
