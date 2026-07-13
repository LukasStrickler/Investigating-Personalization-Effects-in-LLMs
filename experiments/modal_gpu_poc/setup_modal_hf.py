#!/usr/bin/env python3
"""Sync HuggingFace token to Modal for gated subject models.

For gated repos (e.g. Ministral 3, Llama), vLLM needs ``HF_TOKEN`` inside the
Modal container. This script:

  1. Verifies your ``HF_TOKEN`` can access the target model repo
  2. Creates or **replaces** the Modal secret ``huggingface-token`` (used by
     ``modal_serve.py`` when ``MODAL_SERVE_HF_TOKEN=1``). If the secret already
     exists it is deleted and recreated so the token always matches ``.env``.

Usage
    .venv/bin/python experiments/modal_gpu_poc/setup_modal_hf.py
    .venv/bin/python experiments/modal_gpu_poc/setup_modal_hf.py --check-only
    .venv/bin/python experiments/modal_gpu_poc/setup_modal_hf.py --model-id meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_utils import load_dotenv

DEFAULT_MODEL = "mistralai/Ministral-3-8B-Instruct-2512"
SECRET_NAME = "huggingface-token"


def check_hf_access(model_id: str, token: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from None

    api = HfApi(token=token)
    try:
        info = api.model_info(model_id)
    except Exception as exc:
        err = str(exc).lower()
        if "401" in err or "403" in err or "gated" in err or "authorized" in err:
            raise SystemExit(
                f"Cannot access {model_id} with your HF_TOKEN.\n\n"
                "Fix:\n"
                f"  1. Open https://huggingface.co/{model_id}\n"
                "  2. Log in and accept the repository license if prompted\n"
                "  3. Ensure HF_TOKEN in .env is a **Read** token for the same account\n"
                f"  4. Re-run this script\n\n"
                f"Original error: {exc}"
            ) from exc
        raise SystemExit(f"HF API error for {model_id}: {exc}") from exc

    gated = getattr(info, "gated", False) or (info.card_data and info.card_data.get("gated"))
    print(f"OK — HF token can read {model_id}" + (" (gated repo, access granted)" if gated else ""))


def sync_modal_secret(token: str) -> None:
    if not os.environ.get("MODAL_TOKEN_ID"):
        raise SystemExit("MODAL_TOKEN_ID missing — add Modal tokens to .env first.")

    listed = subprocess.run(
        ["modal", "secret", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if SECRET_NAME in (listed.stdout or ""):
        print(f"Updating existing Modal secret {SECRET_NAME!r} …")
        subprocess.run(["modal", "secret", "delete", SECRET_NAME, "--yes"], check=False)

    print(f"Creating Modal secret {SECRET_NAME!r} …")
    proc = subprocess.run(
        ["modal", "secret", "create", SECRET_NAME, f"HF_TOKEN={token}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        proc2 = subprocess.run(
            ["modal", "secret", "create", SECRET_NAME, f"HF_TOKEN={token}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc2.returncode != 0 and "already exists" not in (proc2.stderr or "").lower():
            raise SystemExit(
                f"modal secret create failed:\n{proc2.stderr or proc2.stdout}\n"
                f"Try manually: modal secret create {SECRET_NAME} HF_TOKEN=hf_..."
            )
    print(f"Modal secret {SECRET_NAME!r} ready — deploy with MODAL_SERVE_HF_TOKEN=1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=DEFAULT_MODEL, help="HF repo to verify access for")
    ap.add_argument("--check-only", action="store_true", help="verify HF access only, skip Modal secret")
    args = ap.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN not set in .env\n\n"
            "  1. Accept the model license: https://huggingface.co/" + args.model_id + "\n"
            "  2. Create a Read token: https://huggingface.co/settings/tokens\n"
            "  3. Add HF_TOKEN=hf_... to .env\n"
            "  4. Re-run this script\n"
        )

    check_hf_access(args.model_id, token)
    if not args.check_only:
        sync_modal_secret(token)
    print("\nDeploy gated model with MODAL_SERVE_HF_TOKEN=1 (see modal_serve.py header).")


if __name__ == "__main__":
    main()
