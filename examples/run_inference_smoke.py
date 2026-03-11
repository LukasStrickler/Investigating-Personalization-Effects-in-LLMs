#!/usr/bin/env python3
"""LOW-LEVEL: Run a single inference request for smoke testing.

LAYER NOTE: This is the LOW-LEVEL inference layer (inference package).
For high-level experiment workflows with prompt x model matrices,
see examples/run_experiment_matrix.py.

This script runs one inference request through the shared client.
It's intended for quick validation that the inference pipeline works.

Usage:
    python scripts/run_inference_smoke.py --config config/inference.example.yaml --provider mock
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a single inference request for smoke testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with mock provider (no API key needed)
    python scripts/run_inference_smoke.py --config config/inference.example.yaml --provider mock

    # Run with OpenAI
    python scripts/run_inference_smoke.py --config config/inference.yaml --provider openai
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the inference configuration YAML file.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        help="Provider to use for inference (default: mock).",
    )
    return parser.parse_args()


async def run_smoke(config_path: Path, provider: str) -> int:
    """Run a single smoke test inference request.

    Args:
        config_path: Path to the configuration file.
        provider: Provider name to use.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Import here to avoid loading heavy dependencies at script parse time
    from inference import InferenceRequest, create_client, run_completion

    try:
        # Create client from config
        client = create_client(config_path)

        # Determine model alias based on provider
        model_alias = "mock-test" if provider == "mock" else f"{provider}-default"
        # Create a simple test request
        request = InferenceRequest(
            model_alias=model_alias,
            prompt="Hello, this is a smoke test. Please respond briefly.",
        )

        # Run the completion
        result = await run_completion(client, request)

        # Print result summary
        print("Smoke test completed successfully!")
        print(f"  Provider: {result.provider}")
        print(f"  Model: {result.model}")
        print(f"  Content: {result.content[:100]}{'...' if len(result.content) > 100 else ''}")
        print(f"  Tokens: prompt={result.prompt_tokens}, completion={result.completion_tokens}")
        print(f"  Latency: {result.latency_ms:.1f}ms")

        return 0

    except Exception as e:
        print(f"Smoke test failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    return asyncio.run(run_smoke(args.config, args.provider))


if __name__ == "__main__":
    sys.exit(main())
