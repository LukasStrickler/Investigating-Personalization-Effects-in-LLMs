"""Configuration loading and validation for the inference scaffold.

This module provides:
- Provider registry constants
- YAML configuration loading
- Environment variable resolution for secrets
- Configuration validation

Usage:
    from inference.config import load_config_from_file, resolve_api_key

    config = load_config_from_file("config/inference.yaml")
    api_key = resolve_api_key(config.providers["openai"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from inference.types import InferenceConfig, ProviderConfig

# Provider registry constants
SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "openrouter",
    }
)

MOCK_PROVIDER: str = "mock"

# Test-only providers (not for production use)
TEST_ONLY_PROVIDERS: frozenset[str] = frozenset(
    {
        MOCK_PROVIDER,
    }
)


def is_test_only_provider(provider_name: str) -> bool:
    """Check if a provider is test-only (not for production use).

    Args:
        provider_name: Name of the provider to check.

    Returns:
        True if the provider is test-only, False otherwise.
    """
    return provider_name in TEST_ONLY_PROVIDERS


def load_config(config_dict: dict[str, Any]) -> InferenceConfig:
    """Load and validate configuration from a dictionary.

    Args:
        config_dict: Configuration dictionary (typically from YAML).

    Returns:
        Validated InferenceConfig instance.

    Raises:
        ValidationError: If configuration is invalid.
        ValueError: If provider names are unsupported.
    """
    return InferenceConfig.model_validate(config_dict)


def load_config_from_yaml(yaml_string: str) -> InferenceConfig:
    """Load and validate configuration from a YAML string.

    Args:
        yaml_string: YAML configuration as a string.

    Returns:
        Validated InferenceConfig instance.

    Raises:
        yaml.YAMLError: If YAML parsing fails.
        ValidationError: If configuration is invalid.
    """
    config_dict = yaml.safe_load(yaml_string)
    if config_dict is None:
        config_dict = {}
    return load_config(config_dict)


def load_config_from_file(config_path: str | Path) -> InferenceConfig:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated InferenceConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
        ValidationError: If configuration is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    yaml_content = config_path.read_text(encoding="utf-8")
    return load_config_from_yaml(yaml_content)


def resolve_api_key(provider: ProviderConfig) -> str:
    """Resolve an API key from environment variables.

    Args:
        provider: Provider configuration containing the env var name.

    Returns:
        The API key value from the environment.

    Raises:
        ValueError: If the environment variable is not set.

    Note:
        For the 'mock' test provider, a placeholder key is returned
        if the environment variable is not set.
    """
    env_var_name = provider.api_key_env
    api_key = os.environ.get(env_var_name)

    if api_key is None:
        # Allow mock provider to work without a real key
        if provider.name == MOCK_PROVIDER:
            return f"mock-key-for-{provider.name}"

        raise ValueError(
            f"API key environment variable '{env_var_name}' is not set. "
            f"Please set it before using provider '{provider.name}'."
        )

    return api_key


__all__ = [
    "SUPPORTED_PROVIDERS",
    "MOCK_PROVIDER",
    "TEST_ONLY_PROVIDERS",
    "is_test_only_provider",
    "load_config",
    "load_config_from_yaml",
    "load_config_from_file",
    "resolve_api_key",
]
