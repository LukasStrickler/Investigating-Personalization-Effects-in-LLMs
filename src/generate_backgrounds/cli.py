"""CLI entrypoint for the background generation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).parent


async def _async_main(args: argparse.Namespace) -> int:
    import inference
    from generate_backgrounds.pipeline import BackgroundPipeline, GenerationConfig

    config = GenerationConfig(
        model_alias=args.model_alias,
        mapping_dir=Path(args.mapping_dir),
        output_dir=Path(args.output_dir),
        personas_dir=Path(args.personas_dir),
        concurrency=args.concurrency,
        system_prompt=args.system_prompt,
    )

    client = inference.create_client(args.config)
    pipeline = BackgroundPipeline(client=client, config=config)

    dimensions = args.dimensions if args.dimensions else None

    dimension_results = []
    assembly = None
    error = None

    try:
        dimension_results = await pipeline.run_generation(dimensions)
    except Exception:
        error = traceback.format_exc()

    try:
        assembly = pipeline.assemble_personas()
    except Exception:
        if error is None:
            error = traceback.format_exc()

    # --- Print summary ---
    print()
    print("=== Background Generation Summary ===")
    print()

    any_failures = False
    if dimension_results:
        header = f"{'Dimension':<25} {'Total':>6} {'Generated':>10} {'Skipped':>8} {'Failed':>7}"
        print(header)
        print("-" * len(header))
        for dr in dimension_results:
            print(
                f"{dr.dimension:<25} {dr.total:>6} {dr.generated:>10} {dr.skipped:>8} {dr.failed:>7}"
            )
            if dr.failed:
                any_failures = True
    else:
        print("No dimensions were processed.")

    print()
    if assembly is not None:
        print("=== Persona Assembly Summary ===")
        print(f"  Personas:            {assembly.total_personas}")
        print(f"  Total histories:     {assembly.total_histories}")
        print(f"  Generated histories: {assembly.generated_histories}")
        print(f"  Skipped histories:   {assembly.skipped_histories}")
    else:
        print("=== Persona Assembly Summary ===")
        print("  Skipped (generation did not complete).")
    print()

    if error:
        print("=== Error ===")
        print(error)

    return 1 if (any_failures or error) else 0


def main() -> None:
    default_mapping = str(_HERE / "dimension_value_mapping")
    default_output = str(_HERE / "data" / "backgrounds")
    default_personas = str(_HERE / "data" / "personas")

    parser = argparse.ArgumentParser(
        description="Generate conversation history backgrounds for LLM personalization research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to inference.yaml configuration file",
    )
    parser.add_argument(
        "--model-alias",
        required=True,
        dest="model_alias",
        help="Model alias (defined in inference config) to use for generation",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=None,
        help="Dimensions to generate (default: all discovered). E.g. --dimensions Social_Status",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=default_output,
        help="Output root directory for per-dimension background JSONL files",
    )
    parser.add_argument(
        "--personas-dir",
        dest="personas_dir",
        default=default_personas,
        help="Output directory for assembled persona conversation history JSONL files",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum simultaneous LLM requests",
    )
    parser.add_argument(
        "--mapping-dir",
        dest="mapping_dir",
        default=default_mapping,
        help=(
            "Path to dimension_value_mapping directory. "
            "Use this to point at a test data folder."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt",
        default=None,
        help="Optional system prompt applied to all generation requests",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
