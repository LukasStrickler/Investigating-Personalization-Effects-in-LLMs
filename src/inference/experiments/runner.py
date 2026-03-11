from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.client import InferenceRequest, UnifiedInferenceClient
from inference.experiments.csv_schema import (
    CellStatus,
    MatrixCell,
    canonical_prompt_spec,
    compute_prompt_id,
)
from inference.experiments.dataframe import build_dataframe_from_csv
from inference.experiments.persistence import (
    MatrixCSVWriter,
    build_experiment_csv_path,
    load_existing_matrix,
)
from inference.experiments.scheduling import (
    ExperimentCellStatus,
    ExperimentRetryConfig,
    ExperimentSchedulingConfig,
    RetryAction,
    SchedulingPolicy,
    is_await_all_complete,
    resolve_retry,
)
from inference.experiments.types import ExperimentConfig, ExperimentResult, ExperimentSummary
from inference.logging import CONSOLE_LOG_HEADER, format_completion_console_line
from inference.retry import RetryPolicy


@dataclass(frozen=True, slots=True)
class _PromptEntry:
    row_key: str
    prompt_id: str
    prompt_spec: str | dict


class ExperimentRunner:
    def __init__(self, client: UnifiedInferenceClient) -> None:
        self._client = client

    async def run(self, config: ExperimentConfig) -> ExperimentResult:
        aliases = [alias.strip() for alias in config.model_aliases]
        prompt_entries = _build_prompt_entries(
            config.prompts, default_system_prompt=config.default_system_prompt
        )
        prompt_row_keys = [entry.row_key for entry in prompt_entries]
        completed_cells: dict[tuple[str, str], CellStatus] = {}

        if config.resume_from_existing_csv:
            csv_path = _resolve_resume_path(config)
            writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=aliases)
            _prompt_ids_seen, completed_cells = await asyncio.to_thread(
                load_existing_matrix, csv_path
            )
            await asyncio.to_thread(writer.append_missing_prompts, config.prompts)
            # Trim CSV and sidecar to current prompts so removals are reflected in file and results
            current_prompt_ids = {entry.prompt_id for entry in prompt_entries}
            await asyncio.to_thread(writer.retain_only_prompts, current_prompt_ids)
        else:
            csv_path = build_experiment_csv_path(config.experiment_name)
            writer = MatrixCSVWriter(csv_path=csv_path, model_aliases=aliases)
            await asyncio.to_thread(writer.initialize, config.prompts)
            if config.run_cells is not None:
                for entry in prompt_entries:
                    for alias in aliases:
                        if (entry.prompt_id, alias) not in config.run_cells:
                            await asyncio.to_thread(
                                writer.write_cell,
                                entry.prompt_id,
                                alias,
                                MatrixCell(status=CellStatus.NOT_REQUESTED),
                            )

        retry_config = ExperimentRetryConfig(
            max_retries=config.retry.max_retries,
            max_retry_after_window_seconds=config.scheduling.max_retry_after_wait_seconds,
        )
        scheduling_config = _to_scheduling_config(config)
        retry_policy = RetryPolicy(
            max_retries=config.retry.max_retries,
            base_delay=config.retry.base_delay,
            max_delay=config.retry.max_delay,
            jitter=False,
        )

        status_matrix: dict[str, dict[str, ExperimentCellStatus]] = {
            entry.row_key: dict.fromkeys(aliases, ExperimentCellStatus.PENDING)
            for entry in prompt_entries
        }
        response_matrix: dict[str, dict[str, str | None]] = {
            entry.row_key: dict.fromkeys(aliases, None) for entry in prompt_entries
        }
        error_matrix: dict[str, dict[str, str | None]] = {
            entry.row_key: dict.fromkeys(aliases, None) for entry in prompt_entries
        }

        group_locks = _build_group_locks(aliases, scheduling_config)

        # Build list of (entry, alias) to run; skip NOT_REQUESTED and completed
        run_cells_set = config.run_cells
        cells_to_run: list[tuple[_PromptEntry, str]] = []
        for entry in prompt_entries:
            for alias in aliases:
                if completed_cells.get((entry.prompt_id, alias)) is CellStatus.SUCCESS:
                    status_matrix[entry.row_key][alias] = ExperimentCellStatus.SUCCESS
                    continue
                if run_cells_set is not None and (entry.prompt_id, alias) not in run_cells_set:
                    status_matrix[entry.row_key][alias] = ExperimentCellStatus.NOT_REQUESTED
                    continue
                cells_to_run.append((entry, alias))

        total_cells = len(cells_to_run)

        # Resume existing run: if no cells to run, load CSV and return without calling inference
        if total_cells == 0:
            summary = _build_summary(
                status_matrix, prompt_count=len(config.prompts), model_count=len(aliases)
            )
            dataframe = await asyncio.to_thread(build_dataframe_from_csv, csv_path)
            if config.verbosity != "quiet" and config.resume_from_existing_csv:
                print(
                    "Resume existing run: no cells to run; returning loaded results.",
                    flush=True,
                )
            return ExperimentResult(
                dataframe=dataframe,
                csv_path=csv_path,
                csv_name=csv_path.name,
                summary=summary,
            )

        done_count = 0
        done_lock = asyncio.Lock()

        async def on_cell_done(
            success_result: Any | None = None,
            alias: str = "",
        ) -> None:
            nonlocal done_count
            async with done_lock:
                done_count += 1
                if config.verbosity != "quiet":
                    if success_result is not None:
                        line = format_completion_console_line(
                            provider=getattr(success_result, "provider", "-"),
                            model=getattr(success_result, "model", alias),
                            status="success",
                            latency_ms=getattr(success_result, "latency_ms", None),
                            done_count=done_count,
                            total_cells=total_cells,
                        )
                    else:
                        get_provider = getattr(self._client, "get_provider_model", None)
                        resolved = get_provider(alias) if callable(get_provider) else None
                        if isinstance(resolved, tuple) and len(resolved) == 2:
                            provider, model = resolved
                        else:
                            provider, model = "-", alias
                        line = format_completion_console_line(
                            provider=provider,
                            model=model,
                            status="fail",
                            latency_ms=None,
                            done_count=done_count,
                            total_cells=total_cells,
                        )
                    print(f"  {line}", flush=True)

        if config.verbosity != "quiet":
            label = "Resume existing run" if config.resume_from_existing_csv else "Experiment"
            print(
                f"{label}: {len(prompt_entries)} prompts × {len(aliases)} models = {total_cells} cells",
                flush=True,
            )
            print(f"  {CONSOLE_LOG_HEADER}", flush=True)

        tasks = [
            asyncio.create_task(
                self._run_cell(
                    prompt_entry=entry,
                    alias=alias,
                    status_matrix=status_matrix,
                    response_matrix=response_matrix,
                    error_matrix=error_matrix,
                    csv_writer=writer,
                    retry_policy=retry_policy,
                    retry_config=retry_config,
                    group_lock=group_locks.get(alias),
                    on_cell_done=on_cell_done,
                    system_prompt=config.default_system_prompt,
                    tools=config.tools,
                    tool_choice=config.tool_choice,
                    system_prompt_by_model=config.system_prompt_by_model,
                )
            )
            for entry, alias in cells_to_run
        ]

        await asyncio.gather(*tasks)
        if config.verbosity != "quiet":
            print("Done.", flush=True)

        if not is_await_all_complete(status_matrix, prompt_ids=prompt_row_keys, aliases=aliases):
            raise RuntimeError("Experiment run ended before full matrix reached terminal states.")

        summary = _build_summary(
            status_matrix, prompt_count=len(config.prompts), model_count=len(aliases)
        )
        dataframe = await asyncio.to_thread(build_dataframe_from_csv, csv_path)

        return ExperimentResult(
            dataframe=dataframe,
            csv_path=csv_path,
            csv_name=csv_path.name,
            summary=summary,
        )

    async def _run_cell(
        self,
        *,
        prompt_entry: _PromptEntry,
        alias: str,
        status_matrix: dict[str, dict[str, ExperimentCellStatus]],
        response_matrix: dict[str, dict[str, str | None]],
        error_matrix: dict[str, dict[str, str | None]],
        csv_writer: MatrixCSVWriter,
        retry_policy: RetryPolicy,
        retry_config: ExperimentRetryConfig,
        group_lock: asyncio.Lock | None,
        on_cell_done: Callable[..., Awaitable[None]] | None = None,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        system_prompt_by_model: dict[str, str] | None = None,
    ) -> None:
        row_key = prompt_entry.row_key
        status_matrix[row_key][alias] = ExperimentCellStatus.RUNNING

        attempt = 0
        while True:
            attempt += 1
            try:
                request = _inference_request_from_spec(
                    prompt_entry.prompt_spec,
                    alias,
                    system_prompt,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                if system_prompt_by_model and alias in system_prompt_by_model:
                    request = InferenceRequest(
                        model_alias=request.model_alias,
                        prompt=request.prompt,
                        system_prompt=system_prompt_by_model[alias],
                        messages=request.messages,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )
                result = await self._complete_with_limits(
                    request=request,
                    group_lock=group_lock,
                )
            except Exception as error:
                resolution = resolve_retry(
                    attempt=attempt,
                    error=error,
                    provider=_extract_provider_name(error),
                    retry_policy=retry_policy,
                    config=retry_config,
                    provider_retry_after_seconds=_extract_retry_after_seconds(error),
                )

                if resolution.action in {
                    RetryAction.RETRY_WITH_BACKOFF,
                    RetryAction.RETRY_AFTER_HINT,
                }:
                    await asyncio.sleep(max(0.0, resolution.wait_seconds))
                    continue

                terminal_status = resolution.terminal_status or ExperimentCellStatus.FAILED
                status_matrix[row_key][alias] = terminal_status
                error_matrix[row_key][alias] = str(error)
                await asyncio.to_thread(
                    csv_writer.write_cell,
                    prompt_entry.prompt_id,
                    alias,
                    MatrixCell(
                        status=_to_csv_status(terminal_status),
                        error_message=error_matrix[row_key][alias],
                    ),
                )
                if on_cell_done is not None:
                    await on_cell_done(success_result=None, alias=alias)
                return

            status_matrix[row_key][alias] = ExperimentCellStatus.SUCCESS
            response_matrix[row_key][alias] = result.content
            metadata = getattr(result, "metadata", None)
            await asyncio.to_thread(
                csv_writer.write_cell,
                prompt_entry.prompt_id,
                alias,
                MatrixCell(
                    status=CellStatus.SUCCESS,
                    response=result.content,
                    metadata=metadata,
                ),
            )
            if on_cell_done is not None:
                await on_cell_done(success_result=result, alias=alias)
            return

    async def _complete_with_limits(
        self,
        *,
        request: InferenceRequest,
        group_lock: asyncio.Lock | None,
    ) -> Any:
        if group_lock is not None:
            await group_lock.acquire()
        try:
            return await self._client.complete(request)
        finally:
            if group_lock is not None:
                group_lock.release()


def _inference_request_from_spec(
    spec: str | dict,
    model_alias: str,
    default_system_prompt: str | None,
    *,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> InferenceRequest:
    if isinstance(spec, str):
        return InferenceRequest(
            model_alias=model_alias,
            prompt=spec,
            system_prompt=default_system_prompt,
            tools=tools,
            tool_choice=tool_choice,
        )
    if "messages" in spec:
        return InferenceRequest(
            model_alias=model_alias,
            prompt="",
            system_prompt=spec.get("system"),
            messages=spec["messages"],
            tools=tools,
            tool_choice=tool_choice,
        )
    return InferenceRequest(
        model_alias=model_alias,
        prompt=spec.get("user", ""),
        system_prompt=spec.get("system") or default_system_prompt,
        tools=tools,
        tool_choice=tool_choice,
    )


def _build_prompt_entries(
    prompts: Sequence[str | dict],
    *,
    default_system_prompt: str | None = None,
) -> list[_PromptEntry]:
    entries: list[_PromptEntry] = []
    for index, item in enumerate(prompts):
        if isinstance(item, str) and default_system_prompt:
            prompt_spec: str | dict = {
                "system": default_system_prompt,
                "user": item,
            }
        else:
            prompt_spec = item
        prompt_id = compute_prompt_id(canonical_prompt_spec(prompt_spec))
        row_key = f"{prompt_id}:{index}"
        entries.append(_PromptEntry(row_key=row_key, prompt_id=prompt_id, prompt_spec=prompt_spec))
    return entries


def _resolve_resume_path(config: ExperimentConfig) -> Path:
    if config.existing_csv_path is not None:
        if not config.existing_csv_path.exists():
            raise FileNotFoundError(f"Resume CSV does not exist: {config.existing_csv_path}")
        return config.existing_csv_path

    log_dir = Path("logs") / config.experiment_name
    if not log_dir.exists():
        raise FileNotFoundError(f"No existing experiment logs found: {log_dir}")

    csv_files = sorted(log_dir.glob("*.csv"), reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in experiment logs: {log_dir}")
    return csv_files[0]


def _to_scheduling_config(config: ExperimentConfig) -> ExperimentSchedulingConfig:
    if config.scheduling.interleave_model_aliases:
        return ExperimentSchedulingConfig(policy=SchedulingPolicy.NON_BLOCKING)
    return ExperimentSchedulingConfig(
        policy=SchedulingPolicy.GROUPED,
        alias_group_by_name=dict.fromkeys(config.model_aliases, "default"),
    )


def _build_group_locks(
    aliases: Sequence[str],
    scheduling_config: ExperimentSchedulingConfig,
) -> dict[str, asyncio.Lock]:
    if scheduling_config.policy is SchedulingPolicy.NON_BLOCKING:
        return {}

    lock_by_scope: dict[frozenset[str], asyncio.Lock] = {}
    lock_by_alias: dict[str, asyncio.Lock] = {}
    for alias in aliases:
        scope = scheduling_config.barrier_scope(alias, aliases)
        scope_lock = lock_by_scope.setdefault(scope, asyncio.Lock())
        lock_by_alias[alias] = scope_lock
    return lock_by_alias


def _build_summary(
    status_matrix: Mapping[str, Mapping[str, ExperimentCellStatus]],
    *,
    prompt_count: int,
    model_count: int,
) -> ExperimentSummary:
    statuses = [status for row in status_matrix.values() for status in row.values()]
    terminal_count = sum(1 for status in statuses if status.is_terminal)
    failed_count = sum(1 for status in statuses if status is ExperimentCellStatus.FAILED)
    rate_limited_count = sum(
        1 for status in statuses if status is ExperimentCellStatus.RATE_LIMITED
    )

    return ExperimentSummary(
        prompt_count=prompt_count,
        model_count=model_count,
        total_cells=prompt_count * model_count,
        completed_cells=terminal_count,
        failed_cells=failed_count,
        rate_limited_cells=rate_limited_count,
    )


def _to_csv_status(status: ExperimentCellStatus) -> CellStatus:
    if status is ExperimentCellStatus.SUCCESS:
        return CellStatus.SUCCESS
    if status is ExperimentCellStatus.RATE_LIMITED:
        return CellStatus.RATE_LIMITED
    if status is ExperimentCellStatus.NOT_REQUESTED:
        return CellStatus.NOT_REQUESTED
    return CellStatus.FAILED


def _extract_retry_after_seconds(error: Exception) -> float | None:
    for candidate in _iter_error_chain(error):
        for attr_name in ("retry_after_seconds", "retry_after"):
            parsed = _coerce_wait_seconds(getattr(candidate, attr_name, None))
            if parsed is not None:
                return parsed

        parsed_from_headers = _extract_retry_after_from_headers(getattr(candidate, "headers", None))
        if parsed_from_headers is not None:
            return parsed_from_headers

        parsed_from_message = _extract_retry_after_from_message(str(candidate))
        if parsed_from_message is not None:
            return parsed_from_message

    return None


def _extract_retry_after_from_headers(headers: object) -> float | None:
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if str(key).lower() == "retry-after":
            return _coerce_wait_seconds(value)
    return None


def _extract_retry_after_from_message(message: str) -> float | None:
    patterns = [
        r"retry[-_ ]after\s*[:=]?\s*(?P<seconds>[0-9]+(?:\.[0-9]+)?)",
        r"retry\s+in\s+(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|second|seconds)",
    ]
    lowered = message.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match is not None:
            parsed = _coerce_wait_seconds(match.group("seconds"))
            if parsed is not None:
                return parsed
    return None


def _extract_provider_name(error: Exception) -> str | None:
    for candidate in _iter_error_chain(error):
        provider = getattr(candidate, "provider", None)
        if isinstance(provider, str) and provider.strip():
            return provider

        match = re.search(r"provider=(?:'|\")([^'\"]+)(?:'|\")", str(candidate))
        if match is not None:
            return match.group(1)

    return None


def _iter_error_chain(error: Exception) -> list[Exception]:
    chain: list[Exception] = []
    seen: set[int] = set()
    current: Exception | None = error
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ if isinstance(current.__cause__, Exception) else None
    return chain


def _coerce_wait_seconds(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
    else:
        return None

    if not math.isfinite(parsed):
        return None
    return max(0.0, parsed)


__all__ = ["ExperimentRunner"]
