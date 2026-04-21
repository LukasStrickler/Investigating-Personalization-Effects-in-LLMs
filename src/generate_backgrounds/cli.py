"""CLI entrypoint for the background generation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import traceback
from pathlib import Path

_HERE = Path(__file__).parent


def _make_retry_sleep(bar: "tqdm.tqdm") -> object:  # type: ignore[name-defined]
    """Return an async sleep function that announces retry waits below the progress bar."""
    import asyncio as _asyncio

    async def _sleep(seconds: float) -> None:
        bar.write(f"  ⚠  API retry — waiting {seconds:.1f}s")
        end = _asyncio.get_event_loop().time() + seconds

        async def _tick() -> None:
            while True:
                remaining = end - _asyncio.get_event_loop().time()
                if remaining <= 0.5:
                    break
                bar.set_postfix_str(f"retrying in {remaining:.0f}s")
                await _asyncio.sleep(1)

        tick_task = _asyncio.create_task(_tick())
        await _asyncio.sleep(seconds)
        tick_task.cancel()
        try:
            await tick_task
        except _asyncio.CancelledError:
            pass
        bar.set_postfix_str("")
        bar.write(f"  ✓  Resumed after {seconds:.1f}s wait")

    return _sleep


async def _async_main(args: argparse.Namespace) -> int:
    import tqdm

    from generate_backgrounds.pipeline import BackgroundPipeline, GenerationConfig
    from inference.client import UnifiedInferenceClient
    from inference.providers import _configure_litellm
    from inference.rate_limits import ProviderRateLimiter

    config = GenerationConfig(
        model_alias=args.model_alias,
        mapping_dir=Path(args.mapping_dir),
        output_dir=Path(args.output_dir),
        personas_dir=Path(args.personas_dir),
        concurrency=args.concurrency,
        system_prompt=args.system_prompt,
        verbose=args.verbose,
    )

    dimensions = args.dimensions if args.dimensions else None
    assemble = args.assemble

    # --- Pre-count pending combos for accurate progress bar total ---
    _configure_litellm()
    # Construct with a placeholder sleep; real sleep injected after bar creation
    _sleep_holder: list = []

    async def _proxy_sleep(seconds: float) -> None:
        if _sleep_holder:
            await _sleep_holder[0](seconds)
        else:
            await asyncio.sleep(seconds)

    limiter = ProviderRateLimiter(sleep=_proxy_sleep)
    client = UnifiedInferenceClient.from_config_file(
        args.config, sleep=_proxy_sleep, limiter=limiter,
    )
    pipeline = BackgroundPipeline(client=client, config=config)

    pending_by_dim = pipeline.count_pending(dimensions)
    total_pending = sum(pending_by_dim.values())

    # --- Progress bar ---
    bar = tqdm.tqdm(
        total=total_pending,
        unit="req",
        desc="Generating",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )
    _sleep_holder.append(_make_retry_sleep(bar))

    lock = threading.Lock()
    dim_counts: dict[str, int] = {d: 0 for d in pending_by_dim}
    ok_total = 0
    fail_total = 0

    verbose = getattr(args, "verbose", False)

    def _on_combo_done(dimension: str, record: object) -> None:
        nonlocal ok_total, fail_total
        with lock:
            dim_counts[dimension] = dim_counts.get(dimension, 0) + 1
            done = dim_counts[dimension]
            total_dim = pending_by_dim.get(dimension, "?")
            if record is not None:
                ok_total += 1
            else:
                fail_total += 1
        bar.set_description(f"Generating [{dimension} {done}/{total_dim}]")
        bar.set_postfix_str(f"✓{ok_total} ✗{fail_total}")
        bar.update(1)
        if verbose and record is None:
            bar.write(f"  ✗  [{dimension}] request failed (returned None)")

    dimension_results = []
    assembly = None
    error = None

    try:
        dimension_results = await pipeline.run_generation(
            dimensions, on_combo_done=_on_combo_done
        )
    except Exception:
        error = traceback.format_exc()
    finally:
        bar.close()

    print()
    if assemble:
        print("Assembling personas…", flush=True)
        try:
            import tqdm as _tqdm_mod

            assemble_bar = _tqdm_mod.tqdm(
                unit="hist",
                desc="Assembling",
                dynamic_ncols=True,
            )
            _asm_gen = 0
            _asm_skip = 0

            def _on_total(total: int) -> None:
                assemble_bar.total = total
                assemble_bar.refresh()

            def _on_history(generated: bool) -> None:
                nonlocal _asm_gen, _asm_skip
                if generated:
                    _asm_gen += 1
                else:
                    _asm_skip += 1
                assemble_bar.set_postfix_str(f"new {_asm_gen}, skip {_asm_skip}")
                assemble_bar.update(1)

            assembly = pipeline.assemble_personas(
                on_history_done=_on_history,
                on_total=_on_total,
            )
            assemble_bar.close()
        except Exception:
            if error is None:
                error = traceback.format_exc()
    else:
        print("Skipping persona assembly (use --assemble to enable).", flush=True)

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
        default=1,
        help="Maximum simultaneous LLM requests (default: 1)",
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
        "--assemble",
        dest="assemble",
        action="store_true",
        default=False,
        help="Run persona assembly (Phases 2+3) after background generation.",
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt",
        default=None,
        help="Optional system prompt applied to all generation requests",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed per-request logging (start, finish, errors, retries)",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
