"""Unit tests for the judges package: types, hashing, prompts, parsing, persistence, runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from inference.client import InferenceRequest, InferenceResult, UnifiedInferenceClient
from inference.judges import (
    JUDGE_PROMPT_VERSION,
    NONE_SENTINEL,
    ExperimentDataFrameAdapter,
    GenericRecordsAdapter,
    JudgeConfig,
    JudgeExecutionConfig,
    JudgeLogger,
    JudgeStatus,
    JudgeSubject,
    ParseStatus,
    build_judge_messages,
    judge_config_hash,
    parse_final_answer,
    render_transcript,
    run_judges,
    subjects_from_dataframe,
)
from inference.judges.adapters import _is_missing_scalar
from inference.judges.csv_schema import COLUMNS
from inference.judges.persistence import (
    JudgeCSVCorruptError,
    JudgmentCSVWriter,
)
from inference.judges.runner import _judgment_id
from inference.judges.types import JudgeVerdict

# ----- types / config hash -----


def _cfg(**kw: Any) -> JudgeConfig:
    base: dict[str, Any] = {
        "experiment_name": "t",
        "judges": ["a", "b"],
        "judge_prompt": "Judge the thing.",
        "classes": ["X", "Y"],
    }
    base.update(kw)
    return JudgeConfig(**base)


class TestJudgmentId:
    def test_includes_subject_model_alias(self) -> None:
        h = "abc"
        a = _judgment_id("p1", "alice", "judge", h)
        b = _judgment_id("p1", "bob", "judge", h)
        assert a != b


class TestConfig:
    def test_rejects_empty_experiment_name(self) -> None:
        with pytest.raises(ValueError):
            _cfg(experiment_name="")

    def test_rejects_empty_judges(self) -> None:
        with pytest.raises(ValueError):
            _cfg(judges=[])

    def test_rejects_duplicate_classes(self) -> None:
        with pytest.raises(ValueError):
            _cfg(classes=["X", "X"])

    def test_rejects_reserved_sentinel_as_class(self) -> None:
        with pytest.raises(ValueError):
            _cfg(classes=[NONE_SENTINEL])

    def test_rejects_non_positive_max_tokens(self) -> None:
        with pytest.raises(ValueError):
            _cfg(max_tokens=0)

    def test_rejects_non_positive_thinking_budget(self) -> None:
        with pytest.raises(ValueError):
            _cfg(thinking_budget_tokens=0)
        with pytest.raises(ValueError):
            _cfg(thinking_budget_tokens=-1)

    def test_hash_stable_for_judge_reorder(self) -> None:
        h1 = judge_config_hash(_cfg(judges=["a", "b"]))
        h2 = judge_config_hash(_cfg(judges=["b", "a"]))
        assert h1 == h2

    def test_hash_changes_on_prompt_change(self) -> None:
        h1 = judge_config_hash(_cfg(judge_prompt="P1"))
        h2 = judge_config_hash(_cfg(judge_prompt="P2"))
        assert h1 != h2

    def test_hash_changes_on_class_reorder(self) -> None:
        h1 = judge_config_hash(_cfg(classes=["X", "Y"]))
        h2 = judge_config_hash(_cfg(classes=["Y", "X"]))
        assert h1 != h2

    def test_hash_changes_on_thinking_budget(self) -> None:
        h1 = judge_config_hash(_cfg(thinking_budget_tokens=None))
        h2 = judge_config_hash(_cfg(thinking_budget_tokens=2048))
        assert h1 != h2

    def test_hash_ignores_output_dir_and_resume(self) -> None:
        h1 = judge_config_hash(_cfg(output_dir=None, resume=True))
        h2 = judge_config_hash(_cfg(output_dir=Path("/tmp/a"), resume=False))
        assert h1 == h2

    def test_version_is_present(self) -> None:
        assert isinstance(JUDGE_PROMPT_VERSION, int)
        assert JUDGE_PROMPT_VERSION >= 2


# ----- prompts -----


class TestPrompts:
    def test_transcript_skips_system_and_labels_turns(self) -> None:
        out = render_transcript(
            [
                {"role": "system", "content": "secret"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        assert "secret" not in out
        assert '<turn role="user">hi</turn>' in out
        assert '<turn role="assistant">hello</turn>' in out

    def test_judge_messages_include_class_block_when_classes_set(self) -> None:
        msgs = build_judge_messages(
            _cfg(classes=["A", "B"]),
            JudgeSubject(subject_id="s1", subject_content="hi"),
        )
        assert msgs[0]["role"] == "system"
        assert "<final_answer>LABEL</final_answer>" in msgs[0]["content"]
        assert "- A" in msgs[0]["content"] and "- B" in msgs[0]["content"]
        assert NONE_SENTINEL in msgs[0]["content"]

    def test_judge_messages_enforce_one_call_contract(self) -> None:
        """The single-call contract: tag appears exactly once and is the last line."""
        msgs = build_judge_messages(
            _cfg(classes=["A", "B"]),
            JudgeSubject(subject_id="s1", subject_content="hi"),
        )
        system = msgs[0]["content"]
        assert "EXACTLY ONCE" in system
        assert "LAST line" in system
        assert "No text" in system or "no text" in system.lower()

    def test_judge_messages_no_class_block_when_unset(self) -> None:
        msgs = build_judge_messages(
            _cfg(classes=None),
            JudgeSubject(subject_id="s1", subject_content="hi"),
        )
        assert "<final_answer>" not in msgs[0]["content"]


# ----- parsing -----


class TestParsing:
    def test_free_form_when_no_classes(self) -> None:
        p = parse_final_answer("anything", None)
        assert p.parse_status is ParseStatus.FREE_FORM

    def test_missing_sentinel_when_tag_absent(self) -> None:
        p = parse_final_answer("just prose", ["A"])
        assert p.parse_status is ParseStatus.MISSING_SENTINEL

    def test_matches_last_sentinel(self) -> None:
        raw = (
            "first try <final_answer>WRONG</final_answer>\nactually <final_answer>A</final_answer>"
        )
        p = parse_final_answer(raw, ["A"])
        assert p.final_class == "A"
        assert p.parse_status is ParseStatus.MATCHED

    def test_case_sensitive_strict_no_substring(self) -> None:
        # Lowercase contents do NOT match "A". Tag found but contents invalid → UNMATCHED.
        p = parse_final_answer("<final_answer>a</final_answer>", ["A"])
        assert p.final_class is None
        assert p.parse_status is ParseStatus.UNMATCHED

    def test_no_substring_match_on_reasoning(self) -> None:
        # The reasoning literally contains "A" — must NOT match because no sentinel.
        p = parse_final_answer("My answer is A obviously.", ["A"])
        assert p.final_class is None
        assert p.parse_status is ParseStatus.MISSING_SENTINEL

    def test_none_sentinel_recognised(self) -> None:
        p = parse_final_answer(f"think... <final_answer>{NONE_SENTINEL}</final_answer>", ["A"])
        assert p.none_declared is True
        assert p.parse_status is ParseStatus.NONE_DECLARED

    def test_whitespace_inside_tag_ok(self) -> None:
        p = parse_final_answer("<final_answer>   A   </final_answer>", ["A"])
        assert p.final_class == "A"

    def test_trailing_text_after_sentinel_is_unmatched(self) -> None:
        p = parse_final_answer("<final_answer>A</final_answer>\nextra", ["A"])
        assert p.parse_status is ParseStatus.UNMATCHED
        assert p.final_class is None


# ----- persistence -----


class TestPersistence:
    def test_initialize_creates_file_with_header(self, tmp_path: Path) -> None:
        p = tmp_path / "x.judgments.csv"
        JudgmentCSVWriter(p).initialize()
        assert p.exists()
        with p.open() as f:
            head = f.readline().strip()
        for col in COLUMNS:
            assert col in head
        # Old extraction columns should be gone.
        assert "extraction_call_made" not in head
        assert "extraction_raw_output" not in head

    def test_resume_skips_only_success_rows(self, tmp_path: Path) -> None:

        p = tmp_path / "x.judgments.csv"
        w = JudgmentCSVWriter(p)
        w.initialize()
        v_ok = _vd(JudgeStatus.SUCCESS, "s1", "alice", "h1")
        v_fail = _vd(JudgeStatus.CALL_FAILED, "s2", "alice", "h1")
        w.upsert(v_ok)
        w.upsert(v_fail)

        w2 = JudgmentCSVWriter(p)
        w2.load()
        assert w2.is_completed(
            subject_id="s1", subject_model_alias=None, judge_alias="alice", judge_config_hash="h1"
        )
        assert not w2.is_completed(
            subject_id="s2", subject_model_alias=None, judge_alias="alice", judge_config_hash="h1"
        )

    def test_upsert_replaces_prior_failed_row(self, tmp_path: Path) -> None:
        p = tmp_path / "x.judgments.csv"
        w = JudgmentCSVWriter(p)
        w.initialize()
        w.upsert(_vd(JudgeStatus.CALL_FAILED, "s1", "alice", "h1"))
        w.upsert(_vd(JudgeStatus.SUCCESS, "s1", "alice", "h1"))
        w2 = JudgmentCSVWriter(p)
        w2.load()
        verdicts = w2.all_verdicts()
        assert len(verdicts) == 1
        assert verdicts[0].status is JudgeStatus.SUCCESS

    def test_corrupt_csv_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.judgments.csv"
        p.write_text("not,a,judge,csv\n1,2,3,4\n", encoding="utf-8")
        with pytest.raises(JudgeCSVCorruptError):
            JudgmentCSVWriter(p).load()


def _vd(status: JudgeStatus, sid: str, judge: str, h: str) -> JudgeVerdict:
    return JudgeVerdict(
        judgment_id=f"{sid}-{judge}-{h}",
        subject_id=sid,
        source_id=None,
        prompt_id=None,
        subject_model_alias=None,
        judge_alias=judge,
        judge_config_hash=h,
        status=status,
        raw_output="raw",
        final_class="X" if status is JudgeStatus.SUCCESS else None,
        none_declared=False,
        parse_status=ParseStatus.MATCHED
        if status is JudgeStatus.SUCCESS
        else ParseStatus.MISSING_SENTINEL,
        error_message=None if status is JudgeStatus.SUCCESS else "boom",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_ms=1.0,
        retry_count=0,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        metadata={},
    )


# ----- adapters -----


class TestAdapters:
    def test_generic_adapter_stable_hash_ids(self) -> None:
        recs = [{"content": "a"}, {"content": "b"}]
        a = GenericRecordsAdapter(recs, content_field="content")
        ids1 = [s.subject_id for s in a.iter_subjects()]
        a2 = GenericRecordsAdapter(recs, content_field="content")
        ids2 = [s.subject_id for s in a2.iter_subjects()]
        assert ids1 == ids2
        assert len(set(ids1)) == 2

    def test_generic_adapter_skips_empty_messages(self) -> None:
        recs = [
            {"id": "a", "messages": []},
            {"id": "b", "messages": [{"role": "user", "content": "hi"}]},
        ]
        subs = list(
            GenericRecordsAdapter(recs, id_field="id", messages_field="messages").iter_subjects()
        )
        assert len(subs) == 1
        assert subs[0].subject_id == "b"

    def test_generic_adapter_explicit_id(self) -> None:
        recs = [{"id": "x", "content": "a"}, {"id": "y", "content": "b"}]
        ids = [s.subject_id for s in GenericRecordsAdapter(recs, id_field="id").iter_subjects()]
        assert ids == ["x", "y"]

    def test_is_missing_scalar_handles_non_scalar(self) -> None:
        assert not _is_missing_scalar([1, 2])
        assert not _is_missing_scalar("prompt-1")

    def test_experiment_adapter_skips_nan_prompt_id(self) -> None:
        import pandas as pd

        ok = json.dumps({"status": "success", "response": "hello"})
        df = pd.DataFrame([{"prompt_id": float("nan"), "prompt": "{}", "alice": ok}])
        subs = list(ExperimentDataFrameAdapter(df).iter_subjects())
        assert subs == []

    def test_experiment_adapter_emits_successes_only(self) -> None:
        import pandas as pd

        ok = json.dumps({"status": "success", "response": "hello"})
        bad = json.dumps({"status": "failure", "error_message": "x"})
        df = pd.DataFrame(
            [
                {"prompt_id": "p1", "prompt": "{}", "alice": ok, "bob": bad},
                {"prompt_id": "p2", "prompt": "{}", "alice": bad, "bob": ok},
            ]
        )
        adapter = ExperimentDataFrameAdapter(df)
        subs = list(adapter.iter_subjects())
        assert len(subs) == 2
        keys = {(s.subject_id, s.subject_model_alias) for s in subs}
        assert keys == {("p1", "alice"), ("p2", "bob")}
        s = adapter.summary()
        assert s["subjects_emitted"] == 2
        assert s["skipped_non_success"] == 2


# ----- runner (with a stub client) -----


class StubClient:
    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        """`scripts[model_alias]` is a list of either strings (returned as content)
        or Exception instances (raised). Each call pops the head."""
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self.calls: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def complete(self, request: InferenceRequest) -> InferenceResult:
        async with self._lock:
            script = self._scripts.get(request.model_alias, [])
            if not script:
                raise RuntimeError(f"stub exhausted for {request.model_alias}")
            head = script.pop(0)
        await asyncio.sleep(0)
        self.calls.append(
            {
                "alias": request.model_alias,
                "messages": request.messages,
                "thinking_budget_tokens": request.thinking_budget_tokens,
            }
        )
        if isinstance(head, Exception):
            raise head
        if isinstance(head, tuple):
            delay, content = head
            await asyncio.sleep(delay)
        else:
            content = str(head)
        return InferenceResult(
            model_alias=request.model_alias,
            provider="stub",
            model="stub",
            content=content,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=1.0,
            retry_count=0,
        )


class TestRunner:
    async def _run(
        self,
        scripts,
        classes=None,
        judges=("alice", "bob"),
        subjects=None,
        output_dir=None,
        resume=True,
        exec_cfg=None,
        thinking_budget_tokens=None,
    ):
        client = StubClient(scripts)
        subs = subjects or [
            JudgeSubject(subject_id="s1", subject_content="content one"),
            JudgeSubject(subject_id="s2", subject_content="content two"),
        ]
        cfg = JudgeConfig(
            experiment_name="t",
            judges=list(judges),
            judge_prompt="Judge",
            classes=list(classes) if classes else None,
            output_dir=output_dir,
            resume=resume,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        return (
            client,
            cfg,
            await run_judges(cast(UnifiedInferenceClient, client), subs, cfg, exec_cfg),
        )

    async def test_strict_match_one_call_per_row(self, tmp_path: Path) -> None:
        scripts = {
            "alice": [
                "thinking ...\n<final_answer>A</final_answer>",
                "<final_answer>B</final_answer>",
            ],
            "bob": [
                "<final_answer>B</final_answer>",
                "<final_answer>A</final_answer>",
            ],
        }
        client, _, res = await self._run(scripts, classes=["A", "B"], output_dir=tmp_path)
        df = res.dataframe
        assert len(df) == 4
        assert (df["status"] == JudgeStatus.SUCCESS.value).all()
        assert sorted(df["final_class"].tolist()) == ["A", "A", "B", "B"]
        # Exactly one call per (subject, judge) row — two subjects × two judges = 4 calls.
        per_alias = {"alice": 0, "bob": 0}
        for c in client.calls:
            per_alias[c["alias"]] = per_alias.get(c["alias"], 0) + 1
        assert per_alias == {"alice": 2, "bob": 2}

    async def test_missing_sentinel_marks_classification_failed_no_second_call(
        self, tmp_path: Path
    ) -> None:
        """No extraction fallback: a missing sentinel becomes CLASSIFICATION_FAILED immediately."""
        scripts = {
            "alice": [
                "no sentinel here",  # judge call 1 -> missing sentinel, no retry
                "<final_answer>B</final_answer>",  # judge call 2 -> ok
            ],
            "bob": [
                "<final_answer>A</final_answer>",
                "<final_answer>B</final_answer>",
            ],
        }
        client, _, res = await self._run(scripts, classes=["A", "B"], output_dir=tmp_path)
        df = res.dataframe.sort_values(["judge_alias", "subject_id"]).reset_index(drop=True)

        # Exactly one call per (subject, judge) — no fallback call fired.
        assert len(client.calls) == 4

        alice_rows = df[df["judge_alias"] == "alice"]
        statuses = sorted(alice_rows["status"].tolist())
        assert statuses == sorted(
            [
                JudgeStatus.CLASSIFICATION_FAILED.value,
                JudgeStatus.SUCCESS.value,
            ]
        )
        # Parse status for the miss is MISSING_SENTINEL (no tag at all).
        miss_row = alice_rows[alice_rows["status"] == JudgeStatus.CLASSIFICATION_FAILED.value].iloc[
            0
        ]
        assert miss_row["parse_status"] == ParseStatus.MISSING_SENTINEL.value

    async def test_unmatched_inner_text_marks_classification_failed(self, tmp_path: Path) -> None:
        """Sentinel present but contents not in class list → CLASSIFICATION_FAILED with UNMATCHED."""
        scripts = {
            "alice": ["<final_answer>nope</final_answer>"],
        }
        client, _, res = await self._run(
            scripts,
            classes=["A"],
            judges=("alice",),
            subjects=[JudgeSubject(subject_id="s1", subject_content="x")],
            output_dir=tmp_path,
        )
        df = res.dataframe
        assert df["status"].iloc[0] == JudgeStatus.CLASSIFICATION_FAILED.value
        assert df["parse_status"].iloc[0] == ParseStatus.UNMATCHED.value
        # Still exactly one call — no extraction.
        assert len(client.calls) == 1

    async def test_none_sentinel_is_success(self, tmp_path: Path) -> None:
        scripts = {
            "alice": [
                f"<final_answer>{NONE_SENTINEL}</final_answer>",
                "<final_answer>A</final_answer>",
            ],
        }
        client, _, res = await self._run(
            scripts,
            classes=["A"],
            judges=("alice",),
            output_dir=tmp_path,
        )
        df = res.dataframe
        assert df[df["subject_id"] == "s1"]["none_declared"].iloc[0]
        assert df[df["subject_id"] == "s1"]["status"].iloc[0] == JudgeStatus.SUCCESS.value
        # Two subjects × one judge = exactly two calls.
        assert len(client.calls) == 2

    async def test_one_judge_failure_does_not_block_other(self, tmp_path: Path) -> None:
        scripts = {
            "alice": [RuntimeError("rate limited"), RuntimeError("rate limited")],
            "bob": [
                "<final_answer>A</final_answer>",
                "<final_answer>A</final_answer>",
            ],
        }
        client, _, res = await self._run(scripts, classes=["A"], output_dir=tmp_path)
        df = res.dataframe
        alice = df[df["judge_alias"] == "alice"]
        bob = df[df["judge_alias"] == "bob"]
        assert (alice["status"] == JudgeStatus.CALL_FAILED.value).all()
        assert (bob["status"] == JudgeStatus.SUCCESS.value).all()

    async def test_resume_skips_completed_rows(self, tmp_path: Path) -> None:
        scripts = {
            "alice": ["<final_answer>A</final_answer>", "<final_answer>A</final_answer>"],
            "bob": ["<final_answer>A</final_answer>", "<final_answer>A</final_answer>"],
        }
        client1, cfg1, res1 = await self._run(scripts, classes=["A"], output_dir=tmp_path)
        assert len(client1.calls) == 4

        client2 = StubClient({"alice": [], "bob": []})
        subs = [
            JudgeSubject(subject_id="s1", subject_content="content one"),
            JudgeSubject(subject_id="s2", subject_content="content two"),
        ]
        res2 = await run_judges(cast(UnifiedInferenceClient, client2), subs, cfg1)
        assert len(client2.calls) == 0
        assert len(res2.dataframe) == 4

    async def test_timeout_marks_failed(self, tmp_path: Path) -> None:
        scripts = {
            "alice": [(0.2, "<final_answer>A</final_answer>")],
        }
        client = StubClient(scripts)
        cfg = JudgeConfig(
            experiment_name="t",
            judges=["alice"],
            judge_prompt="Judge",
            classes=["A"],
            output_dir=tmp_path,
        )
        execution = JudgeExecutionConfig(call_timeout_s=0.01)
        res = await run_judges(
            cast(UnifiedInferenceClient, client),
            [JudgeSubject(subject_id="s1", subject_content="x")],
            cfg,
            execution,
        )
        assert res.dataframe["status"].iloc[0] == JudgeStatus.CALL_FAILED.value

    async def test_thinking_budget_tokens_forwarded(self, tmp_path: Path) -> None:
        """When set on JudgeConfig, the runner forwards it on every InferenceRequest."""
        scripts = {
            "alice": ["<final_answer>A</final_answer>", "<final_answer>A</final_answer>"],
        }
        client, _, _ = await self._run(
            scripts,
            classes=["A"],
            judges=("alice",),
            output_dir=tmp_path,
            thinking_budget_tokens=4096,
        )
        assert len(client.calls) == 2
        for c in client.calls:
            assert c["thinking_budget_tokens"] == 4096


# ----- public API smoke -----


def test_public_api_imports() -> None:
    from inference import (
        ExperimentDataFrameAdapter,  # noqa: F401
        GenericRecordsAdapter,  # noqa: F401
        JudgeConfig,  # noqa: F401
        JudgeExecutionConfig,  # noqa: F401
        JudgeLogger,  # noqa: F401
        JudgeResult,  # noqa: F401
        JudgeRunner,  # noqa: F401
        JudgeStatus,  # noqa: F401
        JudgeSubject,  # noqa: F401
        JudgeVerdict,  # noqa: F401
        judge_config_hash,  # noqa: F401
        run_judges,  # noqa: F401
        subjects_from_dataframe,  # noqa: F401
    )


class TestDataFrameInput:
    def test_subjects_from_dataframe(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            [
                {"id": "a", "text": "hi", "label": "X"},
                {"id": "b", "text": "yo", "label": "Y"},
            ]
        )
        subs = subjects_from_dataframe(
            df, id_field="id", content_field="text", metadata_fields=["label"]
        )
        assert [s.subject_id for s in subs] == ["a", "b"]
        assert subs[0].metadata == {"label": "X"}


class TestLogger:
    def test_logger_emits_expected_lines(self) -> None:
        import io

        sink = io.StringIO()
        log = JudgeLogger(verbosity="verbose", sink=sink)
        log.run_start(
            experiment_name="t",
            judges=["a"],
            total_subjects=1,
            skipped_resume=0,
            per_judge_pending={"a": 1},
            csv_path="/tmp/x.csv",
            config_hash="abc123def",
            workers_per_judge={"a": 1},
        )
        log.row_success(
            judge_alias="a",
            subject_id="s1",
            final_class="X",
            none_declared=False,
            latency_ms=12.3,
            total_tokens=100,
        )
        log.judge_done("a")
        log.run_done(
            {
                "per_judge": {"a": {"completed": 1, "classification_failed": 0, "call_failed": 0}},
                "skipped_resume": 0,
            }
        )
        out = sink.getvalue()
        assert "starting run" in out
        assert "s1" in out
        assert "DONE" in out
        assert "SUMMARY" in out

    def test_logger_silent_emits_nothing(self) -> None:
        import io

        sink = io.StringIO()
        log = JudgeLogger(verbosity="silent", sink=sink)
        log.run_start(
            experiment_name="t",
            judges=["a"],
            total_subjects=1,
            skipped_resume=0,
            per_judge_pending={"a": 1},
            csv_path="/tmp/x.csv",
            config_hash="abc",
            workers_per_judge={"a": 1},
        )
        log.row_success(
            judge_alias="a",
            subject_id="s1",
            final_class="X",
            none_declared=False,
            latency_ms=1.0,
            total_tokens=10,
        )
        log.run_done({"per_judge": {}, "skipped_resume": 0})
        assert sink.getvalue() == ""
