#!/usr/bin/env python3
"""LOW-LEVEL batch processing example for the inference library.

LAYER NOTE: This is the LOW-LEVEL inference layer (inference package).
For high-level experiment workflows with prompt x model matrices,
see examples/run_experiment_matrix.py.

This demonstrates how to process multiple requests with checkpoint/resume.
"""

import asyncio
from inference import create_client, run_batch, InferenceRequest


async def generate_requests():
    """Generate a stream of inference requests.

    In a real scenario, you might read these from a file, database, or generate
    them programmatically.
    """
    prompts = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Name a primary color.",
        "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?",
    ]

    for prompt in prompts:
        yield InferenceRequest(model_alias="mock-test", prompt=prompt)


async def main():
    """Run batch processing example."""
    print("=== Batch Processing Example ===\n")

    # Run batch processing
    # This will checkpoint progress and can resume if interrupted
    await run_batch("config/inference.example.yaml", generate_requests())

    print("\n✓ Batch processing completed!")
    print("  - Check logs/inference.jsonl for detailed results")
    print("  - Check checkpoints/batch.jsonl for progress tracking")


if __name__ == "__main__":
    asyncio.run(main())
