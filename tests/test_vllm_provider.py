"""Tests for the additive `vllm` provider.

vLLM exposes an OpenAI-compatible HTTP server. The provider must:
  * route to LiteLLM's "openai/" prefix (NOT "vllm/", which would invoke the
    in-process vllm engine and ignore base_url),
  * forward both `base_url` and `api_base` so the configured endpoint is reached,
  * be a recognized provider name (config + type validation),
  * behave like a single-key provider (no rotation; plain backoff-and-retry).

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

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"


def _vllm_config(log_path: Path) -> InferenceConfig:
    """A minimal vLLM + mock config mirroring config/inference.vllm.example.yaml."""
    return InferenceConfig(
        providers={
            "vllm": ProviderConfig(
                name="vllm",
                api_key_env="VLLM_API_KEY",
                base_url=VLLM_BASE_URL,
                max_concurrency=64,
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
            "mock": ProviderConfig(
                name="mock",
                api_key_env="MOCK_API_KEY",
                rate_limit=RateLimit(requests_per_minute=0, tokens_per_minute=0),
            ),
        },
        default_provider="vllm",
        model_aliases={
            "gemma-4-31b": ModelAliasConfig(
                alias="gemma-4-31b",
                provider="vllm",
                model="gemma-4-31b",
            ),
            "mock-test": ModelAliasConfig(
                alias="mock-test",
                provider="mock",
                model="mock-model",
            ),
        },
        default_retry=RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.02),
        log_path=str(log_path),
    )


# ---------------------------------------------------------------------------
# Routing prefix
# ---------------------------------------------------------------------------


def test_model_string_routes_vllm_to_openai() -> None:
    """vllm provider must emit an `openai/` model string, not `vllm/`."""
    from inference.providers import _model_string

    assert _model_string(provider="vllm", model="gemma-4-31b") == "openai/gemma-4-31b"


def test_model_string_does_not_double_prefix_openai() -> None:
    """An already-`openai/`-prefixed model is left unchanged (idempotent)."""
    from inference.providers import _model_string

    assert _model_string(provider="vllm", model="openai/gemma-4-31b") == "openai/gemma-4-31b"


def test_model_string_identity_routing_unchanged_for_other_providers() -> None:
    """Providers absent from the prefix map keep routing by their own name."""
    from inference.providers import _model_string

    assert _model_string(provider="openai", model="gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _model_string(provider="anthropic", model="claude-3") == "anthropic/claude-3"
    assert _model_string(provider="openrouter", model="x/y") == "openrouter/x/y"


# ---------------------------------------------------------------------------
# Endpoint forwarding (base_url + api_base both reach the call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vllm_call_forwards_endpoint_and_openai_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")

    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    client = UnifiedInferenceClient(
        config=_vllm_config(tmp_path / "inference.jsonl"),
        adapter=LiteLLMProviderAdapter(completion_callable=fake_completion),
    )

    result = await client.complete(InferenceRequest(model_alias="gemma-4-31b", prompt="hi"))

    assert result.content == "ok"
    assert result.provider == "vllm"
    # Routed to the OpenAI-compatible prefix...
    assert captured["model"] == "openai/gemma-4-31b"
    # ...and the configured endpoint is forwarded via BOTH kwargs (with /v1).
    assert captured["base_url"] == VLLM_BASE_URL
    assert captured["api_base"] == VLLM_BASE_URL
    assert captured["base_url"].endswith("/v1")


@pytest.mark.asyncio
async def test_non_vllm_provider_does_not_get_api_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_base is gated to the vllm route; the mock provider never receives it."""
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter

    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return kwargs

    client = UnifiedInferenceClient(
        config=_vllm_config(tmp_path / "inference.jsonl"),
        adapter=LiteLLMProviderAdapter(completion_callable=fake_completion),
    )

    # The mock provider short-circuits before the LiteLLM kwargs path, so the fake
    # completion is never called and no api_base is ever built.
    result = await client.complete(InferenceRequest(model_alias="mock-test", prompt="hi"))

    assert result.provider == "mock"
    assert "api_base" not in captured


# ---------------------------------------------------------------------------
# Provider registration (config + type validation)
# ---------------------------------------------------------------------------


def test_vllm_in_supported_providers() -> None:
    from inference.config import SUPPORTED_PROVIDERS

    assert "vllm" in SUPPORTED_PROVIDERS


def test_vllm_provider_passes_config_validation() -> None:
    """A config with a vllm provider validates (no Unsupported provider error)."""
    from inference.config import load_config_from_yaml

    yaml_text = """
providers:
  vllm:
    name: vllm
    api_key_env: VLLM_API_KEY
    base_url: http://127.0.0.1:8000/v1
model_aliases:
  gemma-4-31b:
    alias: gemma-4-31b
    provider: vllm
    model: gemma-4-31b
default_provider: vllm
"""
    config = load_config_from_yaml(yaml_text)
    assert config.providers["vllm"].name == "vllm"
    assert config.model_aliases["gemma-4-31b"].provider == "vllm"


def test_vllm_provider_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        InferenceConfig(
            providers={
                "vllm": ProviderConfig(
                    name="vllm",
                    api_key_env="VLLM_API_KEY",
                ),
            },
            model_aliases={
                "m": ModelAliasConfig(alias="m", provider="vllm", model="m"),
            },
        )


def test_vllm_base_url_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inference.config import resolve_provider_base_url

    provider = ProviderConfig(
        name="vllm",
        api_key_env="VLLM_API_KEY",
        base_url="http://127.0.0.1:8000/v1",
    )
    assert resolve_provider_base_url(provider) == "http://127.0.0.1:8000/v1"
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9000/v1")
    assert resolve_provider_base_url(provider) == "http://127.0.0.1:9000/v1"


@pytest.mark.asyncio
async def test_vllm_client_uses_vllm_base_url_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.config import resolve_provider_base_url
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9000/v1")

    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    cfg = _vllm_config(tmp_path / "inference.jsonl")
    provider = cfg.providers["vllm"]
    assert resolve_provider_base_url(provider) == "http://127.0.0.1:9000/v1"

    client = UnifiedInferenceClient(
        config=cfg,
        adapter=LiteLLMProviderAdapter(completion_callable=fake_completion),
    )
    await client.complete(InferenceRequest(model_alias="gemma-4-31b", prompt="hi"))
    assert captured["base_url"] == "http://127.0.0.1:9000/v1"
    assert captured["api_base"] == "http://127.0.0.1:9000/v1"


def test_vllm_example_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped config/inference.vllm.example.yaml loads and validates."""
    from inference.config import load_config_from_file

    for key in ("VLLM_API_KEY", "MOCK_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    example = Path(__file__).parent.parent / "config" / "inference.vllm.example.yaml"
    config = load_config_from_file(example)

    assert "vllm" in config.providers
    assert config.providers["vllm"].base_url == VLLM_BASE_URL
    assert config.default_provider == "vllm"
    # The direct-probing experiment column must exist so a naive swap runs unchanged.
    assert config.model_aliases["gemma-4-31b"].provider == "vllm"
    # The served-model-name (alias.model) is what the launcher passes as SERVED=.
    assert config.model_aliases["gemma-4-31b"].model == "gemma-4-31b"


# ---------------------------------------------------------------------------
# Single-key behavior (no rotation; plain backoff-and-retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_vllm_key_does_not_rotate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single VLLM_API_KEY never rotates: the classic backoff-and-retry path applies.

    Mirrors test_single_openrouter_key_preserves_backoff_behavior. Rotation requires
    >1 key AND a :free model; a local vLLM server is neither, so it stays pinned.
    """
    from inference.client import InferenceRequest, UnifiedInferenceClient
    from inference.providers import LiteLLMProviderAdapter

    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")

    keys_used: list[str] = []
    sleep_calls: list[float] = []

    async def sleep_spy(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def completion(**kwargs: Any) -> dict[str, Any]:
        keys_used.append(kwargs["api_key"])
        if len(keys_used) == 1:
            raise RuntimeError("429 rate limit")
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    client = UnifiedInferenceClient(
        config=_vllm_config(tmp_path / "inference.jsonl"),
        adapter=LiteLLMProviderAdapter(completion_callable=completion),
        sleep=sleep_spy,
    )

    result = await client.complete(InferenceRequest(model_alias="gemma-4-31b", prompt="hi"))

    assert result.content == "ok"
    assert keys_used == ["EMPTY", "EMPTY"]  # same key, retried — never rotated
    assert len(sleep_calls) == 1
    assert result.retry_count == 1
