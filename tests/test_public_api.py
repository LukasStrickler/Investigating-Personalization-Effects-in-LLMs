"""Tests for the public API surface of the inference package.

These tests verify that:
1. Research scripts can import helpers without knowing internal module boundaries
2. The public API surface is stable and well-defined
3. Internal modules are NOT accidentally exposed
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ============================================================================
# RED PHASE: Tests that define the expected public API
# ============================================================================


class TestPublicApiSurface:
    """Tests verifying the intended public API surface."""

    def test_top_level_imports_work_without_internal_modules(self) -> None:
        """Research scripts can import core helpers from top-level package."""
        # These imports should work cleanly
        from inference import (
            InferenceConfig,
            InferenceRequest,
            InferenceResult,
            create_client,
            run_batch,
            run_completion,
        )

        # Verify they are the expected types
        assert callable(create_client)
        assert callable(run_completion)
        assert callable(run_batch)
        assert InferenceRequest is not None
        assert InferenceResult is not None
        assert InferenceConfig is not None

    def test_all_exports_explicit_and_limited(self) -> None:
        """The __all__ list explicitly defines the public surface."""
        import inference

        # __all__ must be defined
        assert hasattr(inference, "__all__")

        # Expected public exports (alphabetical)
        expected_exports = {
            "InferenceConfig",
            "InferenceRequest",
            "InferenceResult",
            "__version__",
            "create_client",
            "run_batch",
            "run_completion",
        }

        actual_exports = set(inference.__all__)
        assert actual_exports == expected_exports, (
            f"__all__ mismatch.\n"
            f"  Expected: {sorted(expected_exports)}\n"
            f"  Actual: {sorted(actual_exports)}\n"
            f"  Missing: {expected_exports - actual_exports}\n"
            f"  Extra: {actual_exports - expected_exports}"
        )

    def test_star_import_is_limited_to_all(self) -> None:
        """from inference import * only imports __all__ items."""
        # Create a clean namespace
        namespace: dict[str, object] = {}
        exec("from inference import *", namespace)

        import inference

        # Should only have items from __all__
        # Note: __version__ IS included in __all__ despite starting with _
        # __builtins__ is added by exec() and should be ignored
        expected_names = set(inference.__all__)
        actual_names = {k for k in namespace.keys() if k != "__builtins__"}

        assert actual_names == expected_names, (
            f"Star import leaked extra names.\n"
            f"  Expected: {sorted(expected_names)}\n"
            f"  Actual: {sorted(actual_names)}\n"
            f"  Leaked: {actual_names - expected_names}"
        )


class TestInternalModulesNotExposed:
    """Tests ensuring internal modules are NOT exposed via star imports."""

    def test_providers_not_in_star_import(self) -> None:
        """The 'providers' module should not be imported via star import."""
        namespace: dict[str, object] = {}
        exec("from inference import *", namespace)

        assert "providers" not in namespace, (
            "'providers' should not be imported via 'from inference import *'. "
            "It is an internal implementation detail."
        )

    def test_rate_limits_not_in_star_import(self) -> None:
        """The 'rate_limits' module should not be imported via star import."""
        namespace: dict[str, object] = {}
        exec("from inference import *", namespace)

        assert "rate_limits" not in namespace, (
            "'rate_limits' should not be imported via 'from inference import *'. "
            "It is an internal implementation detail."
        )

    def test_retry_not_in_star_import(self) -> None:
        """The 'retry' module should not be imported via star import."""
        namespace: dict[str, object] = {}
        exec("from inference import *", namespace)

        assert "retry" not in namespace, (
            "'retry' should not be imported via 'from inference import *'. "
            "It is an internal implementation detail."
        )

    def test_logging_not_in_star_import(self) -> None:
        """The 'logging' module should not be imported via star import."""
        namespace: dict[str, object] = {}
        exec("from inference import *", namespace)

        assert "logging" not in namespace, (
            "'logging' should not be imported via 'from inference import *'. "
            "It is an internal implementation detail."
        )

    def test_batch_internals_not_in_star_import(self) -> None:
        """Batch internals should not be imported via star import."""
        namespace: dict[str, object] = {}
        exec("from inference import *", namespace)

        # BatchRunner is internal; users should use run_batch() helper
        assert "BatchRunner" not in namespace, (
            "'BatchRunner' should not be imported via 'from inference import *'. "
            "Use 'run_batch()' helper instead."
        )
        assert "CheckpointEntry" not in namespace, (
            "'CheckpointEntry' should not be imported via 'from inference import *'."
        )
        assert "CheckpointStatus" not in namespace, (
            "'CheckpointStatus' should not be imported via 'from inference import *'."
        )

class TestCreateClientHelper:
    """Tests for the create_client factory function."""

    def test_create_client_signature_accepts_path_or_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_client accepts both str and Path for config_path."""
        from inference import create_client

        # Create a minimal config file
        config_content = """
providers:
  mock:
    name: mock
    api_key_env: MOCK_API_KEY
model_aliases:
  test-alias:
    alias: test-alias
    provider: mock
    model: mock-model
"""
        config_path = tmp_path / "inference.yaml"
        config_path.write_text(config_content)

        monkeypatch.setenv("MOCK_API_KEY", "test-key")

        # Both str and Path should work
        client_from_str = create_client(str(config_path))
        client_from_path = create_client(config_path)

        assert client_from_str is not None
        assert client_from_path is not None

    def test_create_client_returns_usable_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_client returns a client with a complete method."""
        from inference import InferenceRequest, create_client

        config_content = """
providers:
  mock:
    name: mock
    api_key_env: MOCK_API_KEY
model_aliases:
  test-alias:
    alias: test-alias
    provider: mock
    model: mock-model
"""
        config_path = tmp_path / "inference.yaml"
        config_path.write_text(config_content)

        monkeypatch.setenv("MOCK_API_KEY", "test-key")

        client = create_client(config_path)

        # Client should have a complete method (protocol check)
        assert hasattr(client, "complete")
        assert callable(getattr(client, "complete"))

class TestRunCompletionHelper:
    """Tests for the run_completion helper function."""

    @pytest.mark.asyncio
    async def test_run_completion_signature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_completion accepts client and request, returns result."""
        from inference import InferenceRequest, create_client, run_completion

        config_content = """
providers:
  mock:
    name: mock
    api_key_env: MOCK_API_KEY
model_aliases:
  test-alias:
    alias: test-alias
    provider: mock
    model: mock-model
"""
        config_path = tmp_path / "inference.yaml"
        config_path.write_text(config_content)

        monkeypatch.setenv("MOCK_API_KEY", "test-key")

        client = create_client(config_path)
        request = InferenceRequest(model_alias="test-alias", prompt="Hello")

        result = await run_completion(client, request)

        assert result.model_alias == "test-alias"
        assert result.content  # Mock provider returns content


class TestRunBatchHelper:
    """Tests for the run_batch helper function."""

    @pytest.mark.asyncio
    async def test_run_batch_signature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_batch accepts config_path and async iterator of requests."""
        from inference import InferenceRequest, run_batch

        # Use absolute paths in config so checkpoint is created in tmp_path
        log_path = tmp_path / "logs" / "test.jsonl"
        checkpoint_dir = tmp_path / "checkpoints"

        config_content = f"""
providers:
  mock:
    name: mock
    api_key_env: MOCK_API_KEY
model_aliases:
  test-alias:
    alias: test-alias
    provider: mock
    model: mock-model
log_path: {log_path}
checkpoint_path: {checkpoint_dir}/
"""
        config_path = tmp_path / "inference.yaml"
        config_path.write_text(config_content)

        monkeypatch.setenv("MOCK_API_KEY", "test-key")

        async def requests() -> AsyncIterator[InferenceRequest]:
            yield InferenceRequest(model_alias="test-alias", prompt="Hello")
            yield InferenceRequest(model_alias="test-alias", prompt="World")

        # run_batch should complete without error
        await run_batch(config_path, requests())

        # Verify checkpoint was created
        assert checkpoint_dir.exists()
        checkpoint_file = checkpoint_dir / "batch.jsonl"
        assert checkpoint_file.exists()

class TestVersionExposed:
    """Tests for package version exposure."""

    def test_version_is_exposed(self) -> None:
        """The package version is accessible from top-level."""
        from inference import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_in_all(self) -> None:
        """__version__ is in __all__."""
        import inference

        assert "__version__" in inference.__all__


class TestLazyLoading:
    """Tests verifying lazy import behavior."""

    def test_package_loads_without_env_or_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Package can be imported without requiring .env or config files."""
        # Clear any cached imports
        modules_to_clear = [k for k in sys.modules if k.startswith("inference")]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Should not raise even without .env
        import inference

        # Basic sanity check
        assert hasattr(inference, "__version__")
