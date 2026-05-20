"""JudgeRunner: independent per-judge async queues over UnifiedInferenceClient.

One LLM call per (subject, judge) row. The judge is instructed (via prompts.py) to
end its reply with a strict ``<final_answer>X</final_answer>`` line; the result is
parsed once and mapped directly to a verdict. There is no extraction fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from inference.client import InferenceRequest, InferenceResult, UnifiedInferenceClient
from inference.judges.adapters import SubjectAdapter, coerce_to_adapter
from inference.judges.log import JudgeLogger
from inference.judges.parsing import parse_final_answer
from inference.judges.persistence import JudgmentCSVWriter, default_csv_path
from inference.judges.prompts import build_judge_messages
from inference.judges.types import (
    JudgeConfig,
    JudgeExecutionConfig,
    JudgeResult,
    JudgeStatus,
    JudgeSubject,
    JudgeVerdict,
    ParseStatus,
    judge_config_hash,
)

logger = logging.getLogger("inference.judges")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _judgment_id(subject_id: str, judge_alias: str, config_hash: str) -> str:
    return hashlib.sha256(
        f"{subject_id}|{judge_alias}|{config_hash}".encode()
    ).hexdigest()[:24]


class JudgeRunner:
    def __init__(
        self,
        client: UnifiedInferenceClient,
        *,
        log: JudgeLogger | None = None,
    ) -> None:
        self._client = client
        self._log_override = log

    async def run(
        self,
        subjects: SubjectAdapter | list[JudgeSubject] | Any,
        config: JudgeConfig,
        execution: JudgeExecutionConfig | None = None,
    ) -> JudgeResult:
        execution = execution or JudgeExecutionConfig()
        adapter = coerce_to_adapter(subjects)
        all_subjects = list(adapter.iter_subjects())
        cfg_hash = judge_config_hash(config)

        run_log = self._log_override or JudgeLogger(verbosity=config.log_verbosity)

        csv_path = default_csv_path(config.experiment_name, config.output_dir)
        writer = JudgmentCSVWriter(csv_path)
        if config.resume:
            writer.load()
        writer.initialize()
        write_lock = asyncio.Lock()

        # Build per-judge work lists, skipping rows already SUCCESS in the CSV.
        per_judge_pending: dict[str, list[JudgeSubject]] = {a: [] for a in config.judges}
        skipped = 0
        for s in all_subjects:
            for judge_alias in config.judges:
                if config.resume and writer.is_completed(
                    subject_id=s.subject_id,
                    subject_model_alias=s.subject_model_alias,
                    judge_alias=judge_alias,
                    judge_config_hash=cfg_hash,
                ):
                    skipped += 1
                    continue
                per_judge_pending[judge_alias].append(s)

        summary_by_judge: dict[str, dict[str, int]] = {
            a: {
                "completed": 0,
                "classification_failed": 0,
                "call_failed": 0,
                "skipped_resume": 0,
            }
            for a in config.judges
        }

        run_log.run_start(
            experiment_name=config.experiment_name,
            judges=list(config.judges),
            total_subjects=len(all_subjects),
            skipped_resume=skipped,
            per_judge_pending={a: len(per_judge_pending[a]) for a in config.judges},
            csv_path=str(csv_path),
            config_hash=cfg_hash,
            workers_per_judge={a: execution.workers_for(a) for a in config.judges},
        )

        async def queue_for_judge(judge_alias: str) -> None:
            pending = per_judge_pending[judge_alias]
            if not pending:
                run_log.judge_queue_empty(judge_alias)
                return
            workers = execution.workers_for(judge_alias)
            queue: asyncio.Queue[JudgeSubject | None] = asyncio.Queue()
            for s in pending:
                queue.put_nowait(s)
            for _ in range(workers):
                queue.put_nowait(None)  # sentinel per worker

            async def worker() -> None:
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        return
                    try:
                        verdict = await self._judge_one(
                            subject=item,
                            judge_alias=judge_alias,
                            config=config,
                            config_hash=cfg_hash,
                            execution=execution,
                        )
                        async with write_lock:
                            writer.upsert(verdict)
                        bucket = summary_by_judge[judge_alias]
                        if verdict.status is JudgeStatus.SUCCESS:
                            bucket["completed"] += 1
                            run_log.row_success(
                                judge_alias=judge_alias,
                                subject_id=verdict.subject_id,
                                final_class=verdict.final_class,
                                none_declared=verdict.none_declared,
                                latency_ms=verdict.latency_ms,
                                total_tokens=verdict.total_tokens,
                            )
                        elif verdict.status is JudgeStatus.CLASSIFICATION_FAILED:
                            bucket["classification_failed"] += 1
                            run_log.row_classification_failed(
                                judge_alias=judge_alias,
                                subject_id=verdict.subject_id,
                                raw_preview=(verdict.raw_output or "")[:200],
                                latency_ms=verdict.latency_ms,
                            )
                        else:
                            bucket["call_failed"] += 1
                            run_log.row_call_failed(
                                judge_alias=judge_alias,
                                subject_id=verdict.subject_id,
                                error=verdict.error_message or "<unknown>",
                                latency_ms=verdict.latency_ms,
                            )
                    except Exception as exc:
                        logger.exception(
                            "judge worker crashed: judge=%s subject=%s",
                            judge_alias,
                            item.subject_id,
                        )
                        run_log.row_call_failed(
                            judge_alias=judge_alias,
                            subject_id=item.subject_id,
                            error=f"worker exception: {type(exc).__name__}: {exc}",
                            latency_ms=0.0,
                        )
                    finally:
                        queue.task_done()

            tasks = [asyncio.create_task(worker()) for _ in range(workers)]
            await queue.join()
            for t in tasks:
                t.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            run_log.judge_done(judge_alias)

        # Independent per-judge queues — one bad judge doesn't take down the others.
        results = await asyncio.gather(
            *(queue_for_judge(a) for a in config.judges),
            return_exceptions=True,
        )
        for alias, r in zip(config.judges, results):
            if isinstance(r, Exception):
                logger.exception("judge queue %s crashed at top level: %r", alias, r)

        verdicts = writer.all_verdicts()
        df = self._build_dataframe(verdicts)
        summary = {
            "total_subjects": len(all_subjects),
            "total_judges": len(config.judges),
            "skipped_resume": skipped,
            "per_judge": summary_by_judge,
            "csv_path": str(csv_path),
            "adapter_summary": adapter.summary(),
        }
        run_log.run_done(summary)
        return JudgeResult(verdicts=verdicts, dataframe=df, csv_path=csv_path, summary=summary)

    async def _judge_one(
        self,
        *,
        subject: JudgeSubject,
        judge_alias: str,
        config: JudgeConfig,
        config_hash: str,
        execution: JudgeExecutionConfig,
    ) -> JudgeVerdict:
        started_at = _now_iso()
        started_perf = asyncio.get_event_loop().time()
        jid = _judgment_id(subject.subject_id, judge_alias, config_hash)

        messages = build_judge_messages(config, subject)
        judge_req = InferenceRequest(
            model_alias=judge_alias,
            prompt="",
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            thinking_budget_tokens=config.thinking_budget_tokens,
        )
        try:
            judge_res: InferenceResult = await asyncio.wait_for(
                self._client.complete(judge_req),
                timeout=execution.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            return self._failed_verdict(
                jid=jid,
                subject=subject,
                judge_alias=judge_alias,
                config_hash=config_hash,
                error=f"judge call timeout after {execution.call_timeout_s}s",
                started_at=started_at,
                latency_ms=(asyncio.get_event_loop().time() - started_perf) * 1000.0,
                exc=exc,
            )
        except Exception as exc:
            return self._failed_verdict(
                jid=jid,
                subject=subject,
                judge_alias=judge_alias,
                config_hash=config_hash,
                error=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
                latency_ms=(asyncio.get_event_loop().time() - started_perf) * 1000.0,
                exc=exc,
            )

        raw_output = judge_res.content
        outcome = parse_final_answer(raw_output, config.classes)

        # Free-form mode (no classes): always success.
        if config.classes is None:
            return self._success_verdict(
                jid, subject, judge_alias, config_hash, raw_output,
                None, False, ParseStatus.FREE_FORM,
                judge_res, started_at,
            )

        if outcome.parse_status is ParseStatus.MATCHED:
            return self._success_verdict(
                jid, subject, judge_alias, config_hash, raw_output,
                outcome.final_class, False, ParseStatus.MATCHED,
                judge_res, started_at,
            )
        if outcome.parse_status is ParseStatus.NONE_DECLARED:
            return self._success_verdict(
                jid, subject, judge_alias, config_hash, raw_output,
                None, True, ParseStatus.NONE_DECLARED,
                judge_res, started_at,
            )

        # No valid sentinel — one-call flow, no extraction fallback.
        error_msg = (
            "judge reply did not match any allowed class"
            if outcome.parse_status is ParseStatus.UNMATCHED
            else "judge reply did not contain a <final_answer> sentinel"
        )
        return JudgeVerdict(
            judgment_id=jid,
            subject_id=subject.subject_id,
            source_id=subject.source_id,
            prompt_id=subject.prompt_id,
            subject_model_alias=subject.subject_model_alias,
            judge_alias=judge_alias,
            judge_config_hash=config_hash,
            status=JudgeStatus.CLASSIFICATION_FAILED,
            raw_output=raw_output,
            final_class=None,
            none_declared=False,
            parse_status=outcome.parse_status,
            error_message=error_msg,
            prompt_tokens=judge_res.prompt_tokens,
            completion_tokens=judge_res.completion_tokens,
            total_tokens=judge_res.total_tokens,
            latency_ms=judge_res.latency_ms,
            retry_count=judge_res.retry_count,
            started_at=started_at,
            completed_at=_now_iso(),
            metadata=dict(subject.metadata or {}),
        )

    @staticmethod
    def _success_verdict(
        jid: str,
        subject: JudgeSubject,
        judge_alias: str,
        config_hash: str,
        raw_output: str,
        final_class: str | None,
        none_declared: bool,
        parse_status: ParseStatus,
        judge_res: InferenceResult,
        started_at: str,
    ) -> JudgeVerdict:
        return JudgeVerdict(
            judgment_id=jid,
            subject_id=subject.subject_id,
            source_id=subject.source_id,
            prompt_id=subject.prompt_id,
            subject_model_alias=subject.subject_model_alias,
            judge_alias=judge_alias,
            judge_config_hash=config_hash,
            status=JudgeStatus.SUCCESS,
            raw_output=raw_output,
            final_class=final_class,
            none_declared=none_declared,
            parse_status=parse_status,
            error_message=None,
            prompt_tokens=judge_res.prompt_tokens,
            completion_tokens=judge_res.completion_tokens,
            total_tokens=judge_res.total_tokens,
            latency_ms=judge_res.latency_ms,
            retry_count=judge_res.retry_count,
            started_at=started_at,
            completed_at=_now_iso(),
            metadata=dict(subject.metadata or {}),
        )

    @staticmethod
    def _failed_verdict(
        *,
        jid: str,
        subject: JudgeSubject,
        judge_alias: str,
        config_hash: str,
        error: str,
        started_at: str,
        latency_ms: float,
        exc: BaseException,  # noqa: ARG004
    ) -> JudgeVerdict:
        return JudgeVerdict(
            judgment_id=jid,
            subject_id=subject.subject_id,
            source_id=subject.source_id,
            prompt_id=subject.prompt_id,
            subject_model_alias=subject.subject_model_alias,
            judge_alias=judge_alias,
            judge_config_hash=config_hash,
            status=JudgeStatus.CALL_FAILED,
            raw_output="",
            final_class=None,
            none_declared=False,
            parse_status=ParseStatus.MISSING_SENTINEL,
            error_message=error,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=latency_ms,
            retry_count=0,
            started_at=started_at,
            completed_at=_now_iso(),
            metadata=dict(subject.metadata or {}),
        )

    @staticmethod
    def _build_dataframe(verdicts: Iterable[JudgeVerdict]) -> Any:
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for v in verdicts:
            base = {
                "judgment_id": v.judgment_id,
                "subject_id": v.subject_id,
                "source_id": v.source_id,
                "prompt_id": v.prompt_id,
                "subject_model_alias": v.subject_model_alias,
                "judge_alias": v.judge_alias,
                "judge_config_hash": v.judge_config_hash,
                "status": v.status.value,
                "raw_output": v.raw_output,
                "final_class": v.final_class,
                "none_declared": v.none_declared,
                "parse_status": v.parse_status.value,
                "error_message": v.error_message,
                "prompt_tokens": v.prompt_tokens,
                "completion_tokens": v.completion_tokens,
                "total_tokens": v.total_tokens,
                "latency_ms": v.latency_ms,
                "retry_count": v.retry_count,
                "started_at": v.started_at,
                "completed_at": v.completed_at,
            }
            for k, val in (v.metadata or {}).items():
                base[f"metadata_{k}"] = val
            rows.append(base)
        return pd.DataFrame(rows)


async def run_judges(
    client: UnifiedInferenceClient,
    subjects: SubjectAdapter | list[JudgeSubject] | Any,
    config: JudgeConfig,
    execution: JudgeExecutionConfig | None = None,
) -> JudgeResult:
    return await JudgeRunner(client).run(subjects, config, execution)


__all__ = ["JudgeRunner", "run_judges"]
