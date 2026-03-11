"""Typed models for inference configuration.

This module defines Pydantic models for:
- Rate limiting configuration
- Provider configuration
- Model aliases
- Retry policies
- Main inference configuration
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RateLimit(BaseModel):
    """Rate limiting configuration for a provider.

    Attributes:
        requests_per_minute: Maximum requests per minute (0 = unlimited).
        tokens_per_minute: Maximum tokens per minute (0 = unlimited).
    """

    model_config = ConfigDict(frozen=True)

    requests_per_minute: int = Field(
        default=0,
        ge=0,
        description="Maximum requests per minute (0 = unlimited)",
    )
    tokens_per_minute: int = Field(
        default=0,
        ge=0,
        description="Maximum tokens per minute (0 = unlimited)",
    )


class RetryConfig(BaseModel):
    """Retry configuration for transient failures.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retry).
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay in seconds between retries.
    """

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts",
    )
    base_delay: float = Field(
        default=1.0,
        gt=0,
        description="Base delay in seconds for exponential backoff",
    )
    max_delay: float = Field(
        default=60.0,
        gt=0,
        description="Maximum delay in seconds between retries",
    )

    @model_validator(mode="after")
    def validate_delays(self) -> RetryConfig:
        """Ensure base_delay <= max_delay."""
        if self.base_delay > self.max_delay:
            raise ValueError(
                f"base_delay ({self.base_delay}) must be <= max_delay ({self.max_delay})"
            )
        return self


# Supported provider names (v1 scope)
ProviderName = Literal["openai", "anthropic", "openrouter", "mock"]


class ProviderConfig(BaseModel):
    """Configuration for a single inference provider.

    Attributes:
        name: Provider identifier (openai, anthropic, openrouter, mock).
        api_key_env: Environment variable name for the API key.
        rate_limit: Optional rate limiting configuration.
        retry: Optional retry configuration override.
        base_url: Optional base URL override (for proxies/custom endpoints).
        default_model: Optional default model for this provider.
    """

    model_config = ConfigDict(frozen=True)

    name: ProviderName = Field(
        description="Provider identifier",
    )
    api_key_env: str = Field(
        description="Environment variable name for the API key",
    )
    rate_limit: RateLimit | None = Field(
        default=None,
        description="Optional rate limiting configuration",
    )
    retry: RetryConfig | None = Field(
        default=None,
        description="Optional retry configuration override",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional base URL override",
    )
    default_model: str | None = Field(
        default=None,
        description="Optional default model for this provider",
    )


class ModelAliasConfig(BaseModel):
    """Configuration for a model alias.

    Model aliases allow research scripts to use short, memorable names
    that map to specific provider/model combinations.

    Attributes:
        alias: Short name for the model (e.g., "gpt4", "claude3").
        provider: Provider to use for this model.
        model: Full model identifier on the provider.
    """

    model_config = ConfigDict(frozen=True)

    alias: str = Field(
        description="Short name for the model",
    )
    provider: ProviderName = Field(
        description="Provider to use for this model",
    )
    model: str = Field(
        description="Full model identifier on the provider",
    )


class InferenceConfig(BaseModel):
    """Main inference configuration.

    This is the top-level configuration model that contains all
    provider settings, model aliases, and runtime options.

    Attributes:
        providers: Dictionary of provider configurations by name.
        default_provider: Name of the default provider to use.
        model_aliases: Dictionary of model alias configurations.
        log_path: Path for structured inference logs (JSONL).
        checkpoint_path: Directory for batch checkpoint files.
        default_retry: Default retry configuration for all providers.
    """

    model_config = ConfigDict(frozen=True)

    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Provider configurations by name",
    )
    default_provider: str | None = Field(
        default=None,
        description="Name of the default provider",
    )
    model_aliases: dict[str, ModelAliasConfig] = Field(
        default_factory=dict,
        description="Model alias configurations",
    )
    log_path: str | None = Field(
        default="logs/inference.jsonl",
        description="Path for structured inference logs",
    )
    checkpoint_path: str | None = Field(
        default="checkpoints/",
        description="Directory for batch checkpoint files",
    )
    default_retry: RetryConfig | None = Field(
        default=None,
        description="Default retry configuration",
    )

    @model_validator(mode="after")
    def validate_default_provider(self) -> InferenceConfig:
        """Ensure default_provider exists in providers if specified."""
        if self.default_provider is not None and self.default_provider not in self.providers:
            raise ValueError(
                f"default_provider '{self.default_provider}' not found in providers"
            )
        return self

    @field_validator("providers")
    @classmethod
    def validate_provider_names(cls, v: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        """Ensure all provider names are supported."""
        supported = {"openai", "anthropic", "openrouter", "mock"}
        for _key, provider in v.items():
            if provider.name not in supported:
                raise ValueError(
                    f"Unsupported provider '{provider.name}'. "
                    f"Supported providers: {sorted(supported)}"
                )
        return v
