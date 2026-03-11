"""Tests for config loading and validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml


class TestConfigTypes:
    """Tests for config type definitions."""

    def test_rate_limit_model_exists(self) -> None:
        """RateLimit model should be importable."""
        from inference.types import RateLimit

        limit = RateLimit(requests_per_minute=60, tokens_per_minute=90000)
        assert limit.requests_per_minute == 60
        assert limit.tokens_per_minute == 90000

    def test_rate_limit_negative_requests_rejected(self) -> None:
        """Negative requests_per_minute should raise validation error."""
        from pydantic import ValidationError

        from inference.types import RateLimit

        with pytest.raises(ValidationError) as exc_info:
            RateLimit(requests_per_minute=-1, tokens_per_minute=1000)
        assert "requests_per_minute" in str(exc_info.value)

    def test_rate_limit_negative_tokens_rejected(self) -> None:
        """Negative tokens_per_minute should raise validation error."""
        from pydantic import ValidationError

        from inference.types import RateLimit

        with pytest.raises(ValidationError) as exc_info:
            RateLimit(requests_per_minute=100, tokens_per_minute=-500)
        assert "tokens_per_minute" in str(exc_info.value)

    def test_rate_limit_zero_allowed(self) -> None:
        """Zero values should be allowed (meaning unlimited)."""
        from inference.types import RateLimit

        limit = RateLimit(requests_per_minute=0, tokens_per_minute=0)
        assert limit.requests_per_minute == 0
        assert limit.tokens_per_minute == 0


class TestProviderConfig:
    """Tests for provider configuration."""

    def test_provider_config_model_exists(self) -> None:
        """ProviderConfig model should be importable."""
        from inference.types import ProviderConfig, RateLimit

        config = ProviderConfig(
            name="openai",
            api_key_env="OPENAI_API_KEY",
            rate_limit=RateLimit(requests_per_minute=60, tokens_per_minute=90000),
        )
        assert config.name == "openai"
        assert config.api_key_env == "OPENAI_API_KEY"

    def test_provider_config_missing_name_rejected(self) -> None:
        """ProviderConfig without name should raise validation error."""
        from typing import Any, cast

        from pydantic import ValidationError

        from inference.types import ProviderConfig

        with pytest.raises(ValidationError) as exc_info:
            cast(Any, ProviderConfig)(api_key_env="TEST_KEY")
        assert "name" in str(exc_info.value)


class TestInferenceConfig:
    """Tests for the main inference configuration."""

    def test_inference_config_model_exists(self) -> None:
        """InferenceConfig model should be importable."""
        from inference.types import InferenceConfig

        config = InferenceConfig(providers={}, default_provider=None)
        assert config.providers == {}

    def test_inference_config_with_providers(self) -> None:
        """InferenceConfig should accept provider configurations."""
        from inference.types import InferenceConfig, ProviderConfig

        providers = {
            "openai": ProviderConfig(
                name="openai",
                api_key_env="OPENAI_API_KEY",
            )
        }
        config = InferenceConfig(providers=providers, default_provider="openai")
        assert "openai" in config.providers
        assert config.default_provider == "openai"


class TestProviderRegistry:
    """Tests for provider registry contract."""

    def test_supported_providers_constant_exists(self) -> None:
        """SUPPORTED_PROVIDERS constant should be defined."""
        from inference.config import SUPPORTED_PROVIDERS

        assert "openai" in SUPPORTED_PROVIDERS
        assert "anthropic" in SUPPORTED_PROVIDERS
        assert "openrouter" in SUPPORTED_PROVIDERS

    def test_mock_provider_is_test_only(self) -> None:
        """Mock provider should be marked as test-only."""
        from inference.config import MOCK_PROVIDER, is_test_only_provider

        assert MOCK_PROVIDER == "mock"
        assert is_test_only_provider("mock") is True
        assert is_test_only_provider("openai") is False

    def test_unsupported_provider_rejected(self) -> None:
        """Unsupported provider names should be rejected."""
        from typing import Any, cast

        from pydantic import ValidationError

        from inference.types import InferenceConfig, ProviderConfig

        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(
                providers={
                    "unsupported": cast(Any, ProviderConfig)(
                        name="unsupported",
                        api_key_env="UNSUPPORTED_KEY",
                    )
                },
                default_provider="unsupported",
            )
        assert "unsupported" in str(exc_info.value).lower()


class TestYAMLLoading:
    """Tests for YAML config loading."""

    @pytest.fixture
    def valid_config_dict(self) -> dict[str, Any]:
        """Return a valid configuration dictionary."""
        return {
            "providers": {
                "openai": {
                    "name": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "rate_limit": {
                        "requests_per_minute": 60,
                        "tokens_per_minute": 90000,
                    },
                },
                "anthropic": {
                    "name": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
                "mock": {
                    "name": "mock",
                    "api_key_env": "MOCK_API_KEY",
                },
            },
            "default_provider": "openai",
            "log_path": "logs/inference.jsonl",
            "checkpoint_path": "checkpoints/",
        }

    def test_load_config_from_dict(self, valid_config_dict: dict[str, Any]) -> None:
        """Config should load from a dictionary."""
        from inference.config import load_config

        config = load_config(valid_config_dict)
        assert "openai" in config.providers
        assert "anthropic" in config.providers
        assert "mock" in config.providers
        assert config.default_provider == "openai"

    def test_load_config_from_yaml_string(self, valid_config_dict: dict[str, Any]) -> None:
        """Config should load from a YAML string."""
        from inference.config import load_config_from_yaml

        yaml_str = yaml.dump(valid_config_dict)
        config = load_config_from_yaml(yaml_str)
        assert "openai" in config.providers
        assert config.default_provider == "openai"

    def test_load_config_from_file(self, valid_config_dict: dict[str, Any], tmp_path: Path) -> None:
        """Config should load from a YAML file."""
        from inference.config import load_config_from_file

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(valid_config_dict))

        config = load_config_from_file(config_file)
        assert "openai" in config.providers

    def test_missing_optional_provider_not_required(self, tmp_path: Path) -> None:
        """Missing optional providers should not break loading."""
        from inference.config import load_config_from_file

        # Config with only one provider
        config_dict = {
            "providers": {
                "openai": {
                    "name": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                },
            },
            "default_provider": "openai",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict))

        config = load_config_from_file(config_file)
        assert "openai" in config.providers
        assert "anthropic" not in config.providers


class TestEnvResolution:
    """Tests for environment variable resolution."""

    def test_resolve_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API keys should be resolved from environment variables."""
        from inference.config import resolve_api_key
        from inference.types import ProviderConfig

        monkeypatch.setenv("TEST_API_KEY", "test-secret-key-123")

        provider = ProviderConfig(name="openai", api_key_env="TEST_API_KEY")

        api_key = resolve_api_key(provider)

        assert api_key == "test-secret-key-123"

    def test_resolve_api_key_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing API key env var should raise clear error."""
        from inference.config import resolve_api_key
        from inference.types import ProviderConfig

        monkeypatch.delenv("MISSING_API_KEY", raising=False)

        provider = ProviderConfig(name="openai", api_key_env="MISSING_API_KEY")


        with pytest.raises(ValueError) as exc_info:
            resolve_api_key(provider)

        assert "MISSING_API_KEY" in str(exc_info.value)

    def test_mock_provider_no_real_key_required(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Mock provider should work without real API keys."""
        from inference.config import load_config_from_file

        # Remove all API key env vars
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        config_dict = {
            "providers": {
                "mock": {
                    "name": "mock",
                    "api_key_env": "MOCK_API_KEY",
                },
            },
            "default_provider": "mock",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict))

        config = load_config_from_file(config_file)
        assert config.default_provider == "mock"


class TestModelAliases:
    """Tests for model alias configuration."""

    def test_model_alias_config_exists(self) -> None:
        """ModelAliasConfig should be defined."""
        from inference.types import ModelAliasConfig

        alias = ModelAliasConfig(
            alias="gpt4",
            provider="openai",
            model="gpt-4-turbo-preview",
        )
        assert alias.alias == "gpt4"
        assert alias.provider == "openai"
        assert alias.model == "gpt-4-turbo-preview"

    def test_model_alias_in_config(self, tmp_path: Path) -> None:
        """Config should support model aliases."""
        from inference.config import load_config_from_file

        config_dict = {
            "providers": {
                "openai": {"name": "openai", "api_key_env": "OPENAI_API_KEY"},
            },
            "model_aliases": {
                "gpt4": {
                    "alias": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4-turbo-preview",
                },
            },
            "default_provider": "openai",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict))

        config = load_config_from_file(config_file)
        assert "gpt4" in config.model_aliases
        assert config.model_aliases["gpt4"].model == "gpt-4-turbo-preview"


class TestRetryConfig:
    """Tests for retry configuration."""

    def test_retry_config_exists(self) -> None:
        """RetryConfig should be defined."""
        from inference.types import RetryConfig

        retry = RetryConfig(max_retries=3, base_delay=1.0, max_delay=60.0)
        assert retry.max_retries == 3
        assert retry.base_delay == 1.0

    def test_retry_config_negative_retries_rejected(self) -> None:
        """Negative max_retries should be rejected."""
        from pydantic import ValidationError

        from inference.types import RetryConfig

        with pytest.raises(ValidationError):
            RetryConfig(max_retries=-1, base_delay=1.0, max_delay=60.0)

    def test_retry_config_zero_allowed(self) -> None:
        """Zero retries should be allowed (no retry)."""
        from inference.types import RetryConfig

        retry = RetryConfig(max_retries=0, base_delay=1.0, max_delay=60.0)
        assert retry.max_retries == 0




class TestExampleConfig:
    """Tests for the example configuration file."""

    @pytest.fixture
    def example_config_path(self) -> Path:
        """Return path to the example config file."""
        return Path(__file__).parent.parent / "config" / "inference.example.yaml"

    def test_example_config_file_exists(self, example_config_path: Path) -> None:
        """Example config file should exist."""
        assert example_config_path.exists(), (
            f"Example config not found at {example_config_path}"
        )

    def test_example_config_loads_without_secrets(
        self, example_config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Example config should load without real API keys (uses env var placeholders)."""
        from inference.config import load_config_from_file

        # Ensure no real API keys are set
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "MOCK_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        # Config should load (validation happens at load time)
        config = load_config_from_file(example_config_path)

        assert config is not None
        assert len(config.providers) == 4  # openai, anthropic, openrouter, mock

    def test_example_config_has_all_supported_providers(
        self, example_config_path: Path
    ) -> None:
        """Example config should include all v1 supported providers."""
        from inference.config import load_config_from_file

        config = load_config_from_file(example_config_path)

        # Check all v1 providers are present
        assert "openai" in config.providers
        assert "anthropic" in config.providers
        assert "openrouter" in config.providers
        assert "mock" in config.providers

    def test_example_config_provider_names_valid(
        self, example_config_path: Path
    ) -> None:
        """All provider names in example config should be valid."""
        from inference.config import load_config_from_file, SUPPORTED_PROVIDERS, TEST_ONLY_PROVIDERS

        config = load_config_from_file(example_config_path)

        all_valid = SUPPORTED_PROVIDERS | TEST_ONLY_PROVIDERS
        for provider_key, provider in config.providers.items():
            assert provider.name in all_valid, (
                f"Provider '{provider_key}' has invalid name '{provider.name}'"
            )

    def test_example_config_has_model_aliases_per_provider(
        self, example_config_path: Path
    ) -> None:
        """Example config should have at least one model alias per provider."""
        from inference.config import load_config_from_file

        config = load_config_from_file(example_config_path)

        # Collect providers that have aliases
        providers_with_aliases: set[str] = set()
        for alias_config in config.model_aliases.values():
            providers_with_aliases.add(alias_config.provider)

        # Each supported provider should have at least one alias
        for provider in ["openai", "anthropic", "openrouter", "mock"]:
            assert provider in providers_with_aliases, (
                f"No model alias defined for provider '{provider}'"
            )

    def test_example_config_has_mock_alias(
        self, example_config_path: Path
    ) -> None:
        """Example config should have an explicit test-only mock alias."""
        from inference.config import load_config_from_file

        config = load_config_from_file(example_config_path)

        # Find at least one mock alias
        mock_aliases = [
            a for a in config.model_aliases.values() if a.provider == "mock"
        ]
        assert len(mock_aliases) >= 1, "No mock alias defined for testing"

    def test_example_config_paths_are_valid(
        self, example_config_path: Path
    ) -> None:
        """Example config paths should be valid relative paths."""
        from inference.config import load_config_from_file

        config = load_config_from_file(example_config_path)

        # Paths should be defined
        assert config.log_path is not None
        assert config.checkpoint_path is not None

        # Paths should be relative (not absolute)
        assert not Path(config.log_path).is_absolute(), (
            f"log_path should be relative, got: {config.log_path}"
        )
        assert not Path(config.checkpoint_path).is_absolute(), (
            f"checkpoint_path should be relative, got: {config.checkpoint_path}"
        )

    def test_example_config_has_default_retry(
        self, example_config_path: Path
    ) -> None:
        """Example config should define default_retry settings."""
        from inference.config import load_config_from_file

        config = load_config_from_file(example_config_path)

        assert config.default_retry is not None
        assert config.default_retry.max_retries >= 0
        assert config.default_retry.base_delay > 0
        assert config.default_retry.max_delay >= config.default_retry.base_delay

    def test_example_config_rate_limits_valid(
        self, example_config_path: Path
    ) -> None:
        """All rate limits in example config should be valid."""
        from inference.config import load_config_from_file

        config = load_config_from_file(example_config_path)

        for provider_key, provider in config.providers.items():
            if provider.rate_limit is not None:
                assert provider.rate_limit.requests_per_minute >= 0, (
                    f"{provider_key}: requests_per_minute must be >= 0"
                )
                assert provider.rate_limit.tokens_per_minute >= 0, (
                    f"{provider_key}: tokens_per_minute must be >= 0"
                )

    def test_example_config_default_provider_set(
        self, example_config_path: Path
    ) -> None:
        """Example config should have a default_provider set."""
        from inference.config import load_config_from_file

        config = load_config_from_file(example_config_path)

        assert config.default_provider is not None
        assert config.default_provider in config.providers

    def test_example_config_uses_env_var_placeholders(
        self, example_config_path: Path
    ) -> None:
        """Example config should use env var names, not actual keys."""
        yaml_content = example_config_path.read_text()

        # Should not contain anything that looks like a real API key
        # Real keys typically start with sk- or have specific patterns
        import re

        # Check for common API key patterns that should NOT be in the example
        suspicious_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
            r"sk-ant-[a-zA-Z0-9]{20,}",  # Anthropic-style keys
            r"sk-or-[a-zA-Z0-9]{20,}",  # OpenRouter-style keys
        ]

        for pattern in suspicious_patterns:
            matches = re.findall(pattern, yaml_content)
            assert len(matches) == 0, (
                f"Example config contains what looks like a real API key: {matches}"
            )

        # Should use env var references
        assert "OPENAI_API_KEY" in yaml_content
        assert "ANTHROPIC_API_KEY" in yaml_content
        assert "OPENROUTER_API_KEY" in yaml_content
