"""
LiteLLM provider adapter.

Configures LiteLLM at import (suppress_debug_info, verbose off, loggers to CRITICAL)
and wraps completion calls to suppress its "Provider List" print() in notebooks.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from inference.config import MOCK_PROVIDER

_litellm_configured = False


@contextlib.contextmanager
def _suppress_litellm_provider_list_print() -> Any:
    """Patch builtins.print so LiteLLM's 'Provider List' lines are dropped (works in Jupyter)."""
    real_print = builtins.print

    def filtered(*args: Any, **kwargs: Any) -> None:
        s = " ".join(str(a) for a in args)
        if "Provider List" in s or "docs.litellm.ai" in s:
            return
        real_print(*args, **kwargs)

    try:
        builtins.print = filtered
        yield
    finally:
        builtins.print = real_print


def _configure_litellm() -> None:
    """Set LiteLLM flags and loggers so debug/Provider List output is suppressed. Idempotent."""
    global _litellm_configured
    if _litellm_configured:
        return
    litellm = importlib.import_module("litellm")
    setattr(litellm, "suppress_debug_info", True)  # noqa: B010
    if not callable(getattr(litellm, "set_verbose", None)):
        setattr(litellm, "set_verbose", False)  # noqa: B010
    if hasattr(litellm, "verbose"):
        setattr(litellm, "verbose", False)  # noqa: B010
    if hasattr(litellm, "_logging") and hasattr(litellm._logging, "disable_debugging"):
        with contextlib.suppress(Exception):
            litellm._logging.disable_debugging()
    for name in (
        "LiteLLM",
        "litellm",
        "litellm.llms",
        "litellm.utils",
        "litellm.litellm_core_utils",
    ):
        log = logging.getLogger(name)
        log.setLevel(logging.CRITICAL)
        log.propagate = False
    try:
        core = importlib.import_module("litellm_core")
        if hasattr(core, "suppress_debug_info"):
            setattr(core, "suppress_debug_info", True)  # noqa: B010
    except ImportError:
        pass
    _litellm_configured = True


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    provider: str
    model: str
    prompt: str
    system_prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
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
    tool_calls: list[dict[str, Any]] | None = None
    """Optional metadata from the provider (e.g. system_prompt_folded + system_prompt when applied)."""
    metadata: dict[str, Any] | None = None


class ProviderAdapter(Protocol):
    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...


class LiteLLMProviderAdapter:
    def __init__(self, completion_callable: Any | None = None) -> None:
        self._completion_callable = completion_callable

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        if request.provider == MOCK_PROVIDER:
            return self._mock_response(request)
        _configure_litellm()
        completion_fn = self._completion_callable or _get_litellm_acompletion()
        model_str = _model_string(provider=request.provider, model=request.model)
        messages = (
            request.messages
            if request.messages is not None
            else _build_messages(request.prompt, request.system_prompt)
        )
        messages, fold_metadata = _messages_for_model(messages, model_str)
        kwargs: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
            "api_key": request.api_key,
            "base_url": request.base_url,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.tools is not None:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        with _suppress_litellm_provider_list_print():
            response = await completion_fn(**kwargs)
        return _to_provider_response(response, metadata=fold_metadata)

    def _mock_response(self, request: ProviderRequest) -> ProviderResponse:
        combined = (request.system_prompt or "") + " " + request.prompt
        n = max(0, len(combined.split()))
        return ProviderResponse(
            content=f"mock-response:{request.prompt}",
            prompt_tokens=n,
            completion_tokens=1,
            total_tokens=n + 1,
            tool_calls=None,
        )


def _get_litellm_acompletion() -> Any:
    _configure_litellm()
    return importlib.import_module("litellm").acompletion


def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, Any]]:
    if system_prompt is None or system_prompt == "":
        return [{"role": "user", "content": prompt}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _model_string(*, provider: str, model: str) -> str:
    prefix = f"{provider}/"
    return model if model.startswith(prefix) else f"{prefix}{model}"


# Models that reject or ignore the "system" role (e.g. Gemma instruction-tuned via OpenRouter).
# For these we fold the system message into the first user message.
_MODELS_WITHOUT_SYSTEM_MESSAGE = (
    "gemma",
    "gemma-2",
    "gemma-3",
    "codegemma",
)


def _model_lacks_system_message(model_string: str) -> bool:
    lower = model_string.lower()
    return any(lower.find(m) != -1 for m in _MODELS_WITHOUT_SYSTEM_MESSAGE)


def _messages_for_model(
    messages: list[dict[str, Any]], model_string: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """If the model does not support system role (e.g. Gemma), fold system into first user message.
    Returns (messages, metadata or None). When folding was applied, metadata has
    system_prompt_folded=True and system_prompt=<text> for traceability in results.
    """
    if not messages or not _model_lacks_system_message(model_string):
        return (messages, None)
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        role = (m.get("role") if isinstance(m, dict) else None) or ""
        content = m.get("content") if isinstance(m, dict) else None
        if role == "system" and content is not None:
            system_parts.append(content if isinstance(content, str) else str(content))
        else:
            rest.append(m)
    if not system_parts:
        return (messages, None)
    system_text = "\n\n".join(system_parts)
    prefix = system_text + "\n\n"
    out: list[dict[str, Any]] = []
    merged_first = False
    for m in rest:
        role = (m.get("role") if isinstance(m, dict) else None) or ""
        content = m.get("content") if isinstance(m, dict) else None
        if role == "user" and not merged_first:
            user_content = (
                content if isinstance(content, str) else str(content) if content is not None else ""
            )
            out.append({"role": "user", "content": prefix + user_content})
            merged_first = True
        else:
            out.append(m)
    if not merged_first:
        out.insert(0, {"role": "user", "content": prefix.strip()})
    metadata: dict[str, Any] = {
        "system_prompt_folded": True,
        "system_prompt": system_text,
    }
    return (out, metadata)


def _to_provider_response(
    response: Any, *, metadata: dict[str, Any] | None = None
) -> ProviderResponse:
    choices = (
        response.get("choices")
        if isinstance(response, dict)
        else getattr(response, "choices", None)
    ) or []
    content = _content_from_choices(choices)
    usage = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    ) or {}
    tool_calls = _tool_calls_from_choices(choices)
    return ProviderResponse(
        content=content,
        prompt_tokens=_int(usage, "prompt_tokens"),
        completion_tokens=_int(usage, "completion_tokens"),
        total_tokens=_int(usage, "total_tokens"),
        tool_calls=tool_calls,
        metadata=metadata,
    )


def _tool_calls_from_choices(choices: list[Any]) -> list[dict[str, Any]] | None:
    if not choices:
        return None
    c = choices[0]
    msg = c.get("message") if isinstance(c, dict) else getattr(c, "message", None)
    if msg is None:
        return None
    raw = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
    if raw is None or not isinstance(raw, list):
        return None
    return [x if isinstance(x, dict) else dict(x) for x in raw]


def _content_from_choices(choices: list[Any]) -> str:
    if not choices:
        return ""
    c = choices[0]
    msg = c.get("message") if isinstance(c, dict) else getattr(c, "message", None)
    if msg is not None:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            return "".join(str(x.get("text", x) if isinstance(x, dict) else x) for x in content)
        return str(content) if content is not None else ""
    text = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
    return str(text) if text is not None else ""


def _int(obj: Any, key: str) -> int | None:
    if obj is None:
        return None
    v = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return int(v) if v is not None else None


# Configure LiteLLM as soon as this module is loaded so it is set before any completion runs
# (e.g. when user runs "from inference.experiments import ..." or "from inference import create_client").
_configure_litellm()

__all__ = [
    "LiteLLMProviderAdapter",
    "ProviderAdapter",
    "ProviderRequest",
    "ProviderResponse",
]
