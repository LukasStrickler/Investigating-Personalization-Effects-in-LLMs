"""Tests for the additive `modal` provider.

Modal hosts a deployed vLLM OpenAI-compatible server (a ``*.modal.run`` URL), so
the provider mirrors ``vllm``: route to LiteLLM's ``openai/`` prefix, forward both
``base_url`` and ``api_base``, be a recognized provider name, and read its URL
from ``MODAL_BASE_URL`` (which changes per deploy).

All laptop-runnable: no GPU, no network — a fake completion callable stands in
for the LiteLLM call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference.types import (
    InferenceConfig,
    ModelAliasConfig,
    ProviderConfig,
    RateLimit,
    RetryConfig,
)

MODAL_BASE_URL = "https://ws--pers-subject-serve-serve.modal.run/v1"


def _modal_config(log_path: Path) -> InferenceConfig:
    """A minimal modal + mock config mirroring config/inference.modal.example.yaml."""
    return InferenceConfig(
        providers={
            "modal": ProviderConfig(
                name="modal",
                api_key_env="MODAL_API_KEY",
                base_url=MODAL_BASE_URL,
                max_concurrency=100,
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
            "mock": ProviderConfig(
                name="mock",
                api_key_env="MOCK_API_KEY",
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
        },
        default_provider="modal",
        model_aliases={
            "gemma-4-e2b_modal": ModelAliasConfig(
                alias="gemma-4-e2b_modal",
                provider="modal",
                model="gemma-4-e2b",
            ),
            "mock-test": ModelAliasConfig(alias="mock-test", provider="mock", model="mock-model"),
        },
        default_retry=RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.02),
        log_path=str(log_path),
    )


def test_model_string_routes_modal_to_openai() -> None:
    """modal provider must emit an `openai/` model string so LiteLLM uses base_url."""
    from inference.providers import _model_string

    assert _model_string(provider="modal", model="gemma-4-e2b") == "openai/gemma-4-e2b"


@pytest.mark.asyncio
async def test_modal_call_forwards_endpoint_and_openai_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.setenv("MODAL_API_KEY", "EMPTY")
    monkeypatch.delenv("MODAL_BASE_URL", raising=False)

    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    client = UnifiedInferenceClient(
        config=_modal_config(tmp_path / "inference.jsonl"),
        adapter=LiteLLMProviderAdapter(completion_callable=fake_completion),
    )

    result = await client.complete(InferenceRequest(model_alias="gemma-4-e2b_modal", prompt="hi"))

    assert result.content == "ok"
    assert result.provider == "modal"
    assert captured["model"] == "openai/gemma-4-e2b"
    assert captured["base_url"] == MODAL_BASE_URL
    assert captured["api_base"] == MODAL_BASE_URL


def test_modal_in_supported_providers() -> None:
    from inference.config import SUPPORTED_PROVIDERS

    assert "modal" in SUPPORTED_PROVIDERS


def test_modal_provider_passes_config_validation() -> None:
    from inference.config import load_config_from_yaml

    yaml_text = """
providers:
  modal:
    name: modal
    api_key_env: MODAL_API_KEY
    base_url: https://ws--app-serve.modal.run/v1
model_aliases:
  gemma-4-e2b_modal:
    alias: gemma-4-e2b_modal
    provider: modal
    model: gemma-4-e2b
default_provider: modal
"""
    config = load_config_from_yaml(yaml_text)
    assert config.providers["modal"].name == "modal"
    assert config.model_aliases["gemma-4-e2b_modal"].provider == "modal"


def test_modal_provider_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        InferenceConfig(
            providers={
                "modal": ProviderConfig(name="modal", api_key_env="MODAL_API_KEY"),
            },
            model_aliases={"m": ModelAliasConfig(alias="m", provider="modal", model="m")},
        )


def test_modal_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from inference.config import resolve_provider_base_url

    provider = ProviderConfig(
        name="modal",
        api_key_env="MODAL_API_KEY",
        base_url="https://placeholder--app-serve.modal.run/v1",
    )
    monkeypatch.delenv("MODAL_BASE_URL", raising=False)
    assert resolve_provider_base_url(provider) == "https://placeholder--app-serve.modal.run/v1"
    monkeypatch.setenv("MODAL_BASE_URL", MODAL_BASE_URL)
    assert resolve_provider_base_url(provider) == MODAL_BASE_URL


def test_modal_example_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped config/inference.modal.example.yaml loads and validates."""
    from inference.config import load_config_from_file

    for key in ("MODAL_API_KEY", "MODAL_BASE_URL", "OPENROUTER_API_KEY", "MOCK_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    example = Path(__file__).parent.parent / "config" / "inference.modal.example.yaml"
    config = load_config_from_file(example)

    assert "modal" in config.providers
    assert config.default_provider == "modal"
    # The subject alias the runner/notebook use, and its served-model-name.
    assert config.model_aliases["gemma-4-e2b_modal"].provider == "modal"
    assert config.model_aliases["gemma-4-e2b_modal"].model == "gemma-4-e2b"
    # The OpenRouter judge alias must be present so Stage 2 runs from the same config.
    assert config.model_aliases["gpt-4o-mini_paid"].provider == "openrouter"
