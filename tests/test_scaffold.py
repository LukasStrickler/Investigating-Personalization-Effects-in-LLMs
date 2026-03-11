"""Test inference package imports and basic functionality."""

# pytest is not needed for these simple tests


class TestPackageImport:
    """Tests for package import and basic structure."""

    def test_import_inference_package(self) -> None:
        """Verify the inference package can be imported."""
        import inference

        assert inference is not None

    def test_package_has_version(self) -> None:
        """Verify the package exposes a version."""
        import inference

        assert hasattr(inference, "__version__")
        assert inference.__version__ == "0.1.0"

    def test_import_without_env_file(self) -> None:
        """Verify package imports successfully even without .env file.

        This tests the lazy-loading requirement: secrets should not be loaded
        at import time.
        """
        import os

        # Ensure no API keys are set
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
            _ = os.environ.pop(key, None)  # noqa: PLW2901

        # Import should still work
        import inference

        assert inference.__version__ == "0.1.0"
