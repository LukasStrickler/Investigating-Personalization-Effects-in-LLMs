#!/usr/bin/env python3
"""High-level experiment example for research workflows.

This demonstrates the inference.experiments layer: run a prompt x model matrix,
persist results to CSV immediately, and return a pandas DataFrame.

LAYER NOTE: This is the HIGH-LEVEL experiments layer (inference.experiments).
For low-level single requests or batch processing, see:
  - examples/smoke_test.py (single request)
  - examples/batch_example.py (batch processing with checkpointing)

The experiments layer is designed for:
  - Running prompt x model matrices (N prompts x M model aliases)
  - Automatic CSV persistence with resume-from-CSV support
  - Aggregate results returned as pandas DataFrames
  - Research workflows requiring systematic model comparisons
"""

from __future__ import annotations

import asyncio

from inference import create_client
from inference.experiments import ExperimentConfig, ExperimentRunner


async def main() -> None:
    """Run a prompt x model experiment matrix and display results."""
    print("=== High-Level Experiment Matrix Example ===\n")

    # Create a low-level client (experiments layer uses this internally)
    client = create_client("config/inference.example.yaml")

    # Define the experiment configuration
    # This creates a 3 prompts x 1 model alias = 3 cell matrix
    # For multi-model comparisons with real providers, use:
    #   model_aliases=["gpt-4o-mini", "claude-3-5-sonnet"]
    config = ExperimentConfig(
        experiment_name="research-comparison-example",
        model_aliases=["mock-test"],  # Using mock for demo (works without API keys)
        prompts=[
            "What is the capital of Germany?",
            "Explain quantum entanglement in one sentence.",
            "List three primary colors.",
        ],
        verbosity="normal",
    )

    print(f"Experiment: {config.experiment_name}")
    print(f"Prompts: {len(config.prompts)}")
    print(f"Model aliases: {config.model_aliases}")
    print(f"Matrix size: {len(config.prompts)} x {len(config.model_aliases)} cells")
    print()

    # Run the experiment matrix
    # The runner will:
    #   1. Create logs/<experiment-name>/<timestamp>.csv
    #   2. Execute all N x M cells with retry/scheduling
    #   3. Persist each cell result immediately
    #   4. Return DataFrame + CSV metadata after full completion
    runner = ExperimentRunner(client=client)
    result = await runner.run(config)

    # Display results
    print("=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Experiment name: {config.experiment_name}")
    print(f"CSV saved to: {result.csv_path}")
    print(f"CSV filename: {result.csv_name}")
    print()
    print("Summary:")
    print(f"  - Prompts: {result.summary.prompt_count}")
    print(f"  - Models: {result.summary.model_count}")
    print(f"  - Total cells: {result.summary.total_cells}")
    print(f"  - Completed: {result.summary.completed_cells}")
    print(f"  - Failed: {result.summary.failed_cells}")
    print(f"  - Rate limited: {result.summary.rate_limited_cells}")
    print()
    print("DataFrame preview:")
    print(result.dataframe)

    print("\n✓ Experiment matrix completed!")
    print(f"  - Full results available at: {result.csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
