"""
Inference Scaffold - Unified LLM inference for research.

This package provides a thin wrapper around LiteLLM for multi-provider
inference with rate limiting, retries, structured logging, and batch execution.

Example usage:
    from inference import create_client, run_completion

    client = create_client("config/inference.yaml")
    result = await run_completion(client, request)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Lazy imports to avoid loading config/env at import time
# The public API is populated via __getattr__ for lazy loading

# Type-only imports for type checkers (not executed at runtime)
if TYPE_CHECKING:
    from inference.client import InferenceRequest, InferenceResult, UnifiedInferenceClient
    from inference.types import InferenceConfig


def __getattr__(name: str) -> object:
    """Lazy import public API items on first access.

    This pattern allows the package to be imported without loading
    heavy dependencies (pydantic, litellm) until actually needed.
    """
    # Types from inference.client
    if name == "InferenceRequest":
        from inference.client import InferenceRequest

        return InferenceRequest
    if name == "InferenceResult":
        from inference.client import InferenceResult

        return InferenceResult

    # Types from inference.types
    if name == "InferenceConfig":
        from inference.types import InferenceConfig

        return InferenceConfig

    # Helper functions
    if name == "create_client":
        return _create_client
    if name == "run_completion":
        return _run_completion
    if name == "run_batch":
        return _run_batch

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_client(config_path: str | Path) -> UnifiedInferenceClient:
    """Create a UnifiedInferenceClient from a configuration file.

    This is the primary factory function for creating inference clients.
    Research scripts can use this to get a configured client without
    knowing internal module boundaries.

    Args:
        config_path: Path to the YAML configuration file.
            Can be a string or Path object.

    Returns:
        A configured UnifiedInferenceClient ready for inference calls.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValidationError: If the configuration is invalid.

    Example:
        >>> client = create_client("config/inference.yaml")
        >>> result = await client.complete(request)
    """
    from inference.client import UnifiedInferenceClient

    return UnifiedInferenceClient.from_config_file(config_path)


# Assign to module-level name for direct access
_create_client = create_client


async def run_completion(
    client: UnifiedInferenceClient,
    request: InferenceRequest,
) -> InferenceResult:
    """Execute a single inference request using a client.

    This helper provides a simple function interface for single requests.
    For batch processing, use run_batch() instead.

    Args:
        client: A UnifiedInferenceClient instance (from create_client()).
        request: The inference request to execute.

    Returns:
        The inference result with content, tokens, and metadata.

    Raises:
        InferenceRequestError: If the request fails after retries.
        UnknownModelAliasError: If the model_alias is not configured.

    Example:
        >>> from inference import create_client, run_completion, InferenceRequest
        >>> client = create_client("config/inference.yaml")
        >>> request = InferenceRequest(model_alias="gpt4", prompt="Hello!")
        >>> result = await run_completion(client, request)
        >>> print(result.content)
    """
    return await client.complete(request)


# Assign to module-level name for direct access
_run_completion = run_completion


async def run_batch(
    config_path: str | Path,
    requests: AsyncIterator[InferenceRequest],
) -> None:
    """Execute a batch of inference requests with checkpointing.

    This helper creates a client and batch runner internally, handling
    checkpoint persistence for resumable batch processing.

    Args:
        config_path: Path to the YAML configuration file.
        requests: An async iterator yielding InferenceRequest objects.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        BatchCheckpointError: If the checkpoint file is corrupt.

    Note:
        Progress is checkpointed after each request. If interrupted,
        re-running with the same config will skip completed requests.

    Example:
        >>> from inference import InferenceRequest, run_batch
        >>> async def my_requests():
        ...     yield InferenceRequest(model_alias="gpt4", prompt="Hello")
        ...     yield InferenceRequest(model_alias="gpt4", prompt="World")
        >>> await run_batch("config/inference.yaml", my_requests())
    """
    from inference.batch import BatchRunner
    from inference.client import UnifiedInferenceClient
    from inference.logging import InferenceLogger

    config_path = Path(config_path)
    client = UnifiedInferenceClient.from_config_file(config_path)

    # Load config to get log_path
    from inference.config import load_config_from_file

    config = load_config_from_file(config_path)

    # Determine log path (default to logs/inference.jsonl if not specified)
    log_path = Path(config.log_path) if config.log_path else Path("logs/inference.jsonl")
    logger = InferenceLogger(log_file=log_path)

    # Determine checkpoint path
    checkpoint_path = (
        Path(config.checkpoint_path) / "batch.jsonl"
        if config.checkpoint_path
        else Path("checkpoints/batch.jsonl")
    )

    runner = BatchRunner(client=client, logger=logger, checkpoint_path=checkpoint_path)
    await runner.run_batch(requests)

# Assign to module-level name for direct access
_run_batch = run_batch


# Explicit public API surface
# These are the ONLY names that will be imported with "from inference import *"
__all__ = [
    "__version__",
    "create_client",
    "run_completion",
    "run_batch",
    "InferenceRequest",
    "InferenceResult",
    "InferenceConfig",
]
