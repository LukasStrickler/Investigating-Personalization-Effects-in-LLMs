from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

from inference.config import MOCK_PROVIDER


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    provider: str
    model: str
    prompt: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ProviderAdapter(Protocol):
    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...


class LiteLLMProviderAdapter:
    def __init__(self, completion_callable: Any | None = None) -> None:
        self._completion_callable = completion_callable

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        if request.provider == MOCK_PROVIDER:
            return self._mock_response(request)

        completion_callable = self._completion_callable or _load_litellm_completion()
        response = await completion_callable(
            model=_resolve_litellm_model(provider=request.provider, model=request.model),
            messages=[{"role": "user", "content": request.prompt}],
            api_key=request.api_key,
            base_url=request.base_url,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return _normalize_response(response)

    def _mock_response(self, request: ProviderRequest) -> ProviderResponse:
        prompt_tokens = len(request.prompt.split())
        completion_tokens = 1
        return ProviderResponse(
            content=f"mock-response:{request.prompt}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


def _load_litellm_completion() -> Any:
    litellm = importlib.import_module("litellm")
    return litellm.acompletion


def _resolve_litellm_model(*, provider: str, model: str) -> str:
    provider_prefix = f"{provider}/"
    if model.startswith(provider_prefix):
        return model
    return f"{provider_prefix}{model}"


def _normalize_response(response: Any) -> ProviderResponse:
    choices = _get_value(response, "choices") or []
    content = _extract_content(choices)
    usage = _get_value(response, "usage")

    prompt_tokens = _get_int(usage, "prompt_tokens")
    completion_tokens = _get_int(usage, "completion_tokens")
    total_tokens = _get_int(usage, "total_tokens")

    return ProviderResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _extract_content(choices: list[Any]) -> str:
    if not choices:
        return ""

    first_choice = choices[0]
    message = _get_value(first_choice, "message")
    if message is not None:
        content = _get_value(message, "content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    text = _get_value(first_choice, "text")
    if text is None:
        return ""
    return str(text)


def _get_value(container: Any, key: str) -> Any | None:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _get_int(container: Any, key: str) -> int | None:
    value = _get_value(container, key)
    if value is None:
        return None
    return int(value)


__all__ = [
    "LiteLLMProviderAdapter",
    "ProviderAdapter",
    "ProviderRequest",
    "ProviderResponse",
]
