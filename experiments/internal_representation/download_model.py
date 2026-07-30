#!/usr/bin/env python3
"""Download the default Gemma model into this folder for fully local runs."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--output", help="Local target directory (default: models/<model-name>)")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    target = (
        Path(args.output or Path(__file__).parent / "models" / args.model.rsplit("/", 1)[-1])
        .expanduser()
        .resolve()
    )
    snapshot_download(repo_id=args.model, local_dir=target)
    print(f"Model downloaded to {target}")
    print(f"Run: python main.py --model {target} --attributes Gender")


if __name__ == "__main__":
    main()
