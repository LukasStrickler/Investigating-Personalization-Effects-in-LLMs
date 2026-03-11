#!/usr/bin/env python3
"""LOW-LEVEL: Run batch inference from a JSONL input file.

LAYER NOTE: This is the LOW-LEVEL inference layer (inference package).
For high-level experiment workflows with prompt x model matrices,
see examples/run_experiment_matrix.py.

This script reads inference requests from a JSONL file and processes them
using the shared batch runner with checkpointing for resumability.

Usage:
    python scripts/run_inference_batch.py --config config/inference.example.yaml --input requests.jsonl --provider mock

Input JSONL format (one JSON object per line):
    {"model_alias": "mock-test", "prompt": "Hello, world!"}
    {"model_alias": "mock-test", "prompt": "Test prompt 2"}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference from a JSONL input file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input JSONL format:
    Each line should be a JSON object with:
    - model_alias: The model alias to use (defined in config)
    - prompt: The prompt text
    - max_tokens (optional): Maximum tokens to generate
    - temperature (optional): Sampling temperature

Examples:
    # Run batch with mock provider
    python scripts/run_inference_batch.py --config config/inference.example.yaml --input requests.jsonl --provider mock

    # Run batch with OpenAI
    python scripts/run_inference_batch.py --config config/inference.yaml --input requests.jsonl --provider openai
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the inference configuration YAML file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the JSONL file containing inference requests.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        help="Provider to use for inference (default: mock).",
    )
    return parser.parse_args()


async def read_requests_from_jsonl(input_path: Path) -> AsyncIterator:
    """Read inference requests from a JSONL file.

    Args:
        input_path: Path to the JSONL input file.

    Yields:
        InferenceRequest objects.
    """
    from inference import InferenceRequest

    with open(input_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_number}: {e}", file=sys.stderr)
                continue

            # Extract fields from JSON
            model_alias = data.get("model_alias")
            prompt = data.get("prompt")

            if not model_alias or not prompt:
                print(
                    f"Warning: Skipping line {line_number}: missing model_alias or prompt",
                    file=sys.stderr,
                )
                continue

            # Create request with optional parameters
            kwargs = {
                "model_alias": model_alias,
                "prompt": prompt,
            }
            if "max_tokens" in data:
                kwargs["max_tokens"] = data["max_tokens"]
            if "temperature" in data:
                kwargs["temperature"] = data["temperature"]

            yield InferenceRequest(**kwargs)


async def run_batch(config_path: Path, input_path: Path, provider: str) -> int:
    """Run batch inference from a JSONL file.

    Args:
        config_path: Path to the configuration file.
        input_path: Path to the JSONL input file.
        provider: Provider name to use.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from inference import run_batch

    try:
        # Read requests from file
        requests = read_requests_from_jsonl(input_path)

        # Count total requests for progress reporting
        with open(input_path, encoding="utf-8") as f:
            total_count = sum(1 for line in f if line.strip())
        print(f"Processing {total_count} requests from {input_path}...")
        print(f"Provider: {provider}")
        print(f"Config: {config_path}")
        print()

        # Run the batch
        await run_batch(config_path, requests)

        print()
        print("Batch processing completed successfully!")
        print("Check logs/ and checkpoints/ directories for output files.")

        return 0

    except Exception as e:
        print(f"Batch processing failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    return asyncio.run(run_batch(args.config, args.input, args.provider))


if __name__ == "__main__":
    sys.exit(main())
