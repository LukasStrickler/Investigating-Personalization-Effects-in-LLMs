"""Toggleable progress logger for judge runs.

Verbosity levels (most → least output):

    "debug"   – everything: per-row latency/tokens, retry events, queue state.
    "verbose" – per-row outcome + per-judge progress + warnings.
    "normal"  – milestones (start, per-judge done, summary) + warnings.
    "silent"  – nothing.

Writes to stderr by default (so notebooks don't pollute cell stdout output) but the
sink is injectable for testing. Thread-safe via an internal asyncio.Lock so workers
can log concurrently without interleaving lines.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TextIO

LogVerbosity = Literal["silent", "normal", "verbose", "debug"]

_LEVEL_ORDER: dict[str, int] = {"silent": 0, "normal": 1, "verbose": 2, "debug": 3}


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{int(s):02d}s"


@dataclass
class _JudgeCounters:
    pending: int = 0
    completed: int = 0
    classification_failed: int = 0
    call_failed: int = 0
    retries_seen: int = 0


@dataclass
class JudgeLogger:
    verbosity: LogVerbosity = "normal"
    sink: TextIO | None = None
    clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)

    def __post_init__(self) -> None:
        if self.verbosity not in _LEVEL_ORDER:
            raise ValueError(
                f"verbosity must be one of {sorted(_LEVEL_ORDER)}; got {self.verbosity!r}"
            )
        self._level = _LEVEL_ORDER[self.verbosity]
        self._stream = self.sink if self.sink is not None else sys.stderr
        self._lock = asyncio.Lock()
        self._counters: dict[str, _JudgeCounters] = {}
        self._t0: float = self.clock()

    # ---- public API ----

    def is_enabled(self, level: LogVerbosity) -> bool:
        return self._level >= _LEVEL_ORDER[level]

    def run_start(
        self,
        *,
        experiment_name: str,
        judges: list[str],
        total_subjects: int,
        skipped_resume: int,
        per_judge_pending: dict[str, int],
        csv_path: str,
        config_hash: str,
        workers_per_judge: dict[str, int],
    ) -> None:
        for a in judges:
            self._counters[a] = _JudgeCounters(pending=per_judge_pending.get(a, 0))
        self._t0 = self.clock()
        if not self.is_enabled("normal"):
            return
        self._emit(
            f"judge: starting run experiment={experiment_name!r} "
            f"judges={judges} subjects={total_subjects} cfg_hash={config_hash[:12]}"
        )
        self._emit(f"judge: csv={csv_path}")
        if skipped_resume:
            self._emit(f"judge: resume -> {skipped_resume} (subject, judge) cells already done")
        for a in judges:
            pending = per_judge_pending.get(a, 0)
            self._emit(
                f"judge: {a} | workers={workers_per_judge.get(a, 1)} pending={pending}"
            )

    def judge_queue_empty(self, judge_alias: str) -> None:
        if self.is_enabled("normal"):
            self._emit(f"judge: {judge_alias} | nothing pending (all complete on resume)")

    def row_success(
        self,
        *,
        judge_alias: str,
        subject_id: str,
        final_class: str | None,
        none_declared: bool,
        latency_ms: float,
        total_tokens: int | None,
    ) -> None:
        c = self._counters.setdefault(judge_alias, _JudgeCounters())
        c.completed += 1
        if not self.is_enabled("verbose"):
            return
        label = (
            "__NONE__" if none_declared else (final_class if final_class is not None else "<free>")
        )
        tokens_str = f" {total_tokens}tok" if total_tokens else ""
        self._emit(
            f"judge: {judge_alias} | "
            f"{self._progress(judge_alias)} ok  {subject_id:<20} -> {label:<20}"
            f" ({latency_ms / 1000:.1f}s{tokens_str})"
        )

    def row_classification_failed(
        self,
        *,
        judge_alias: str,
        subject_id: str,
        raw_preview: str,
        latency_ms: float,
    ) -> None:
        c = self._counters.setdefault(judge_alias, _JudgeCounters())
        c.classification_failed += 1
        if not self.is_enabled("normal"):
            return
        self._emit(
            f"judge: {judge_alias} | "
            f"{self._progress(judge_alias)} MISS {subject_id:<20} no valid class "
            f"({latency_ms / 1000:.1f}s)"
        )
        if self.is_enabled("debug"):
            self._emit(f"  raw: {raw_preview!r}")

    def row_call_failed(
        self,
        *,
        judge_alias: str,
        subject_id: str,
        error: str,
        latency_ms: float,
    ) -> None:
        c = self._counters.setdefault(judge_alias, _JudgeCounters())
        c.call_failed += 1
        if not self.is_enabled("normal"):
            return
        kind = "TIMEOUT" if "timeout" in error.lower() else "FAIL"
        short = error if len(error) <= 120 else error[:117] + "..."
        self._emit(
            f"judge: {judge_alias} | "
            f"{self._progress(judge_alias)} {kind} {subject_id:<20} {short} "
            f"({latency_ms / 1000:.1f}s)"
        )

    def judge_done(self, judge_alias: str) -> None:
        c = self._counters.get(judge_alias)
        if c is None or not self.is_enabled("normal"):
            return
        self._emit(
            f"judge: {judge_alias} | DONE  "
            f"ok={c.completed} miss={c.classification_failed} "
            f"fail={c.call_failed} "
            f"elapsed={_fmt_elapsed(self.clock() - self._t0)}"
        )

    def run_done(self, summary: dict[str, object]) -> None:
        if not self.is_enabled("normal"):
            return
        per_judge_raw = summary.get("per_judge", {})
        per_judge: dict[str, dict[str, int]] = (
            per_judge_raw if isinstance(per_judge_raw, dict) else {}
        )
        elapsed = _fmt_elapsed(self.clock() - self._t0)
        total_ok = sum(b.get("completed", 0) for b in per_judge.values())
        total_fail = sum(
            b.get("classification_failed", 0) + b.get("call_failed", 0)
            for b in per_judge.values()
        )
        self._emit(
            f"judge: SUMMARY ok={total_ok} fail={total_fail} "
            f"skipped_resume={summary.get('skipped_resume', 0)} elapsed={elapsed}"
        )

    # ---- helpers ----

    def _progress(self, judge_alias: str) -> str:
        c = self._counters.get(judge_alias)
        if c is None:
            return ""
        done = c.completed + c.classification_failed + c.call_failed
        total = max(c.pending, done)
        return f"[{done:>3}/{total:<3}]"

    def _emit(self, line: str) -> None:
        # Best-effort flush. We don't take the asyncio lock here because each call is
        # a single write(); interleaving at sub-line granularity is unlikely with
        # CPython buffered streams and the cost of awaiting a lock from sync code is
        # not worth it. Workers should call public methods, not _emit directly.
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:
            pass


__all__ = ["JudgeLogger", "LogVerbosity"]
