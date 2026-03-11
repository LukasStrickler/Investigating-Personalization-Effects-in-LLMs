#!/usr/bin/env python3
"""LOW-LEVEL smoke test for the inference library.

LAYER NOTE: This is the LOW-LEVEL inference layer (inference package).
For high-level experiment workflows with prompt x model matrices,
see examples/run_experiment_matrix.py.

This demonstrates the most basic usage: run a single inference request.
"""

import asyncio
from inference import create_client, run_completion, InferenceRequest


async def main():
    """Run a single smoke test request."""
    # Create client from config file
    client = create_client("config/inference.example.yaml")

    # Create a simple request
    request = InferenceRequest(
        model_alias="mock-test", prompt="Hello, this is a smoke test. Please respond briefly."
    )

    # Run the completion
    result = await run_completion(client, request)

    # Print the result
    print("=== Smoke Test Result ===")
    print(f"Provider: {result.provider}")
    print(f"Model: {result.model}")
    print(f"Content: {result.content}")
    print(
        f"Tokens: {result.total_tokens} (prompt={result.prompt_tokens}, completion={result.completion_tokens})"
    )
    print(f"Latency: {result.latency_ms:.1f}ms")
    print("\n✓ Smoke test passed!")


if __name__ == "__main__":
    asyncio.run(main())
