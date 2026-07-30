"""Tests for scripts/run_cluster_direct_probing_stage2.py CLI contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_cluster_direct_probing_stage2.py"


def _load_cli():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("run_cluster_direct_probing_stage2", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_cli()


def test_help_lists_csv_path(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        rv.main(["--help"])
    assert exc.value.code == 0
    assert "--csv-path" in capsys.readouterr().out


def test_missing_csv_returns_nonzero(tmp_path: Path) -> None:
    code = rv.main(
        [
            "--config",
            str(tmp_path / "missing.yaml"),
            "--csv-path",
            str(tmp_path / "missing.csv"),
            "--model-alias",
            "m",
            "--judge-alias",
            "j",
        ]
    )
    assert code == 2


def test_vllm_config_rejected_for_stage2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = Path(__file__).parent.parent / "config" / "inference.vllm.example.yaml"
    csv_path = tmp_path / "matrix.csv"
    csv_path.write_text("prompt_id,prompt\n", encoding="utf-8")
    code = rv.main(
        [
            "--config",
            str(config),
            "--csv-path",
            str(csv_path),
            "--model-alias",
            "gemma-4-31b",
            "--judge-alias",
            "gemma-3-4b",
        ]
    )
    assert code == 2
    assert "OpenRouter" in capsys.readouterr().err


def test_expected_verdict_count_scales_with_judges() -> None:
    assert rv._expected_verdict_count(n_subjects=10, n_judges=1) == 10
    assert rv._expected_verdict_count(n_subjects=10, n_judges=2) == 20
