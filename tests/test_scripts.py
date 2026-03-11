"""Tests for run_inference_smoke.py and run_inference_batch.py scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Path to examples (moved from scripts/)
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
SMOKE_SCRIPT = EXAMPLES_DIR / "run_inference_smoke.py"
BATCH_SCRIPT = EXAMPLES_DIR / "run_inference_batch.py"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "inference.example.yaml"

# Scripts were consolidated into notebooks; skip script tests when scripts are absent
skip_if_no_smoke = pytest.mark.skipif(not SMOKE_SCRIPT.exists(), reason="run_inference_smoke.py removed (use notebooks)")
skip_if_no_batch = pytest.mark.skipif(not BATCH_SCRIPT.exists(), reason="run_inference_batch.py removed (use notebooks)")


class TestSmokeScript:
    """Tests for run_inference_smoke.py script."""

    @skip_if_no_smoke
    def test_script_exists(self) -> None:
        """Smoke script file exists."""
        assert SMOKE_SCRIPT.exists(), f"Smoke script not found at {SMOKE_SCRIPT}"

    @skip_if_no_smoke
    def test_help_exits_cleanly(self) -> None:
        """Smoke script --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--config" in result.stdout, "Missing --config in help"
        assert "--provider" in result.stdout, "Missing --provider in help"

    @skip_if_no_smoke
    def test_smoke_script_succeeds_with_mock_provider(self, tmp_path: Path) -> None:
        """Smoke script runs one mocked inference request through the shared client."""
        result = subprocess.run(
            [
                sys.executable,
                str(SMOKE_SCRIPT),
                "--config",
                str(CONFIG_PATH),
                "--provider",
                "mock",
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )

        assert result.returncode == 0, f"Smoke script failed: {result.stderr}"
        assert (
            "content" in result.stdout.lower()
            or "ok" in result.stdout.lower()
            or "success" in result.stdout.lower()
        ), f"Expected result summary in stdout, got: {result.stdout}"

    @skip_if_no_smoke
    def test_smoke_script_fails_without_config(self) -> None:
        """Smoke script exits non-zero when config is missing."""
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT), "--provider", "mock"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Script should fail without --config"


class TestBatchScript:
    """Tests for run_inference_batch.py script."""

    @skip_if_no_batch
    def test_script_exists(self) -> None:
        """Batch script file exists."""
        assert BATCH_SCRIPT.exists(), f"Batch script not found at {BATCH_SCRIPT}"

    @skip_if_no_batch
    def test_help_exits_cleanly(self) -> None:
        """Batch script --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, str(BATCH_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--config" in result.stdout, "Missing --config in help"
        assert "--input" in result.stdout, "Missing --input in help"
        assert "--provider" in result.stdout, "Missing --provider in help"

    @skip_if_no_batch
    def test_batch_script_rejects_missing_input_path(self, tmp_path: Path) -> None:
        """Batch script exits non-zero with clear error for missing input file."""
        nonexistent_input = tmp_path / "does-not-exist.jsonl"

        result = subprocess.run(
            [
                sys.executable,
                str(BATCH_SCRIPT),
                "--config",
                str(CONFIG_PATH),
                "--input",
                str(nonexistent_input),
                "--provider",
                "mock",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Script should fail with missing input"
        error_combined = (result.stderr + result.stdout).lower()
        assert (
            "error" in error_combined
            or "not found" in error_combined
            or "does not exist" in error_combined
        ), f"Expected clear error message, got: {result.stderr}\n{result.stdout}"

    @skip_if_no_batch
    def test_batch_script_processes_fixture_input(self, tmp_path: Path) -> None:
        """Batch script reads fixture JSONL input and writes outputs/checkpoints/logs."""
        fixture_input = FIXTURES_DIR / "batch_requests.jsonl"
        assert fixture_input.exists(), f"Fixture not found: {fixture_input}"

        result = subprocess.run(
            [
                sys.executable,
                str(BATCH_SCRIPT),
                "--config",
                str(CONFIG_PATH),
                "--input",
                str(fixture_input),
                "--provider",
                "mock",
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )

        assert result.returncode == 0, f"Batch script failed: {result.stderr}"
        output = result.stdout.lower()
        assert (
            "success" in output or "complete" in output or "processed" in output or "ok" in output
        ), f"Expected progress summary in stdout, got: {result.stdout}"

        # Verify log file was created (default path from config)
        log_path = tmp_path / "logs" / "inference.jsonl"
        assert log_path.exists(), f"Log file not created at {log_path}"

        # Verify checkpoint was created (default path from config)
        checkpoint_path = tmp_path / "checkpoints" / "batch.jsonl"
        assert checkpoint_path.exists(), f"Checkpoint file not created at {checkpoint_path}"

    def test_batch_script_fails_without_required_args(self) -> None:
        """Batch script exits non-zero when required args are missing."""
        # Missing --input
        result = subprocess.run(
            [sys.executable, str(BATCH_SCRIPT), "--config", str(CONFIG_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Script should fail without --input"

        # Missing --config
        result = subprocess.run(
            [
                sys.executable,
                str(BATCH_SCRIPT),
                "--input",
                str(FIXTURES_DIR / "batch_requests.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Script should fail without --config"
