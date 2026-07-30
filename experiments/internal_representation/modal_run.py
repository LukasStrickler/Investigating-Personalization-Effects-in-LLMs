#!/usr/bin/env python3
"""Run the internal-representation persona probing pipeline on a Modal GPU.

WHY THIS EXISTS
    ``main.py`` runs the probe locally, but the large open-weights models this
    pipeline is most interesting for — e.g. ``google/gemma-4-31B-it`` (~61 GB
    bf16) or the ``gemma-4-26B-A4B`` MoE (~50 GB) — do not fit in a laptop's
    memory. This wraps the *unchanged* pipeline (``run_pipeline`` in ``main.py``)
    in a Modal batch function that loads the model on a cloud GPU, extracts
    hidden states, trains the probes, and streams the ``results/`` + ``plots/``
    artifacts back to the caller's machine.

    It reuses the same conventions as ``scripts/modal/modal_serve.py``:
    the shared HF cache volume ``pers-hf-cache`` and the ``huggingface-token``
    secret (for gated Gemma-4 repos). Sync that secret first with the existing
    helper:

        .venv/bin/python scripts/modal/setup_modal_hf.py \\
            --model-id google/gemma-4-31B-it

SMOKE TEST FIRST (cheap — small model, cheap GPU)
    MODAL_IR_GPU=L4 modal run experiments/internal_representation/modal_run.py \\
        --model google/gemma-4-E2B-it --samples 8

FULL RUN (the 31B dense model needs an 80 GB GPU)
    MODAL_IR_GPU=A100-80GB modal run experiments/internal_representation/modal_run.py \\
        --model google/gemma-4-31B-it --samples 40 --dtype bfloat16

INDICATOR ABLATION (Gender-only; ablates gender + region indicators)
    # Probe + ablate in one container, capped at 200 F/200 M for cost:
    MODAL_IR_GPU=A100-80GB modal run experiments/internal_representation/modal_run.py \\
        --model google/gemma-4-31B-it --ablation --ablation-subset 200

    # Ablate a PRIOR run without re-probing (reuses its probe/hidden/dataset
    # from the volume). The source run must have been produced AFTER this
    # feature was added — older runs (e.g. results_modal_gemma4_31b) lack the
    # persisted .npz + data/ and must be re-run first:
    MODAL_IR_GPU=A100-80GB modal run experiments/internal_representation/modal_run.py \\
        --ablation-only --from-run results_modal --model google/gemma-4-31B-it \\
        --ablation-subset 200

Artifacts land in ``experiments/internal_representation/<out>/{results,plots,data}``
(default ``out=results_modal``), so local runs in ``results/`` are never
clobbered.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Config (deploy/run-time env knobs)
# --------------------------------------------------------------------------- #
APP_NAME = os.getenv("MODAL_IR_APP_NAME", "pers-internal-repr-probe")
GPU = os.getenv("MODAL_IR_GPU", "A100-80GB")  # L4 / A10G / L40S / A100-80GB / H100
TIMEOUT = int(os.getenv("MODAL_IR_TIMEOUT", str(6 * 60 * 60)))
HF_CACHE = "/cache/hf"
HF_VOLUME = os.getenv("MODAL_IR_HF_VOLUME", "pers-hf-cache")  # shared with modal_serve.py
HF_TOKEN_SECRET = os.getenv("MODAL_IR_HF_TOKEN_SECRET", "huggingface-token")

# Repo-relative paths (this file lives in experiments/internal_representation/).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PERSONAS_LOCAL = _REPO_ROOT / "src" / "generate_backgrounds" / "data" / "personas" / "personas.jsonl"
_MAPPING_DIR_LOCAL = _REPO_ROOT / "src" / "generate_backgrounds" / "dimension_value_mapping"

# Container paths.
_PIPELINE_DIR = "/app/pipeline"          # the bundled internal_representation modules
_PERSONAS_REMOTE = "/app/personas.jsonl"  # bundled dataset (no repo checkout needed in-container)
# Indicator mapping CSVs bundled for the ablation stage (gender + region).
_GENDER_CSV_REMOTE = "/app/mappings/gender.csv"
_REGION_CSV_REMOTE = "/app/mappings/region.csv"
_ABLATION_MAPPINGS = [_GENDER_CSV_REMOTE, _REGION_CSV_REMOTE]

image = (
    modal.Image.debian_slim(python_version="3.12")
    # requirements.txt installs transformers from a git URL, so pip needs `git`.
    .apt_install("git")
    .pip_install_from_requirements(str(_HERE / "requirements.txt"))
    .pip_install("huggingface_hub[hf_transfer]")
    # Mistral/Ministral tokenizers may need mistral_common for the slow-tokenizer
    # fallback in extraction.py; harmless for other models.
    .pip_install("mistral_common>=1.11.0")
    # FP8-quantized checkpoints (e.g. Ministral-3-8B) run their matmuls through
    # transformers' finegrained-fp8 path, which needs the `kernels` package.
    .pip_install("kernels>=0.15.2,<0.16.0")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Bundle the pipeline code + dataset INTO the image. Using add_local_* means
    # local edits are picked up on each `modal run` without a rebuild.
    # NOTE: Modal's ignore patterns are exact/glob — a bare "results_modal" only
    # matches a dir literally named that, NOT "results_modal_gemma4_31b_ablation".
    # Use trailing-* globs so the large downloaded run dirs (incl. multi-GB .npz)
    # are actually excluded. neutral_data/ is intentionally NOT ignored — the
    # NEUTRAL_GENERATED condition reads its bundled dataset.
    .add_local_dir(
        str(_HERE),
        _PIPELINE_DIR,
        ignore=["models", ".venv", "results", "results/*", "plots", "plots/*",
                "results_modal*", "run_results*", "final_run_200*",
                "neutral_backgrounds*", "neutral_personas*",
                "__pycache__", "**/__pycache__", "*.pyc", "*.npz"],
    )
    .add_local_file(str(_PERSONAS_LOCAL), _PERSONAS_REMOTE)
    # Indicator mappings for the ablation stage (Gender + Region).
    .add_local_file(str(_MAPPING_DIR_LOCAL / "gender.csv"), _GENDER_CSV_REMOTE)
    .add_local_file(str(_MAPPING_DIR_LOCAL / "region.csv"), _REGION_CSV_REMOTE)
)

app = modal.App(APP_NAME, image=image)
hf_cache_vol = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)
# Results are also persisted to this volume so a `modal run --detach` job keeps
# its output even if the local client disconnects. Fetch with:
#   modal volume get pers-ir-results <run_name>/ ./results_modal_gemma4_31b
OUTPUT_VOLUME = os.getenv("MODAL_IR_OUTPUT_VOLUME", "pers-ir-results")
OUTPUT_MOUNT = "/output"
output_vol = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)


def _hf_secrets() -> list[modal.Secret]:
    """Attach the HF token secret if it exists (needed for gated Gemma-4 repos)."""
    try:
        return [modal.Secret.from_name(HF_TOKEN_SECRET)]
    except Exception:  # noqa: BLE001 — secret not created; public models still work
        return []


@app.function(
    gpu=GPU,
    volumes={HF_CACHE: hf_cache_vol, OUTPUT_MOUNT: output_vol},
    secrets=_hf_secrets(),
    timeout=TIMEOUT,
)
def run_probe(
    model_id: str,
    attributes: list[str],
    samples: int | None,
    dtype: str,
    max_seq_length: int,
    token_position: str,
    context_mode: str,
    include_partial: bool,
    run_name: str,
    ablation: bool = False,
    ablation_subset: int = 0,
    ablation_only: bool = False,
    from_run: str = "",
    batch_size: int = 4,
) -> dict[str, bytes]:
    """Run the probing pipeline (and optionally indicator ablation) on the GPU.

    Persists artifacts to the ``pers-ir-results`` volume under ``<run_name>/``
    (so a detached run survives client disconnects) AND returns them as a
    mapping ``{"results/<file>": bytes, "plots/<file>": bytes, ...}`` for the
    local entrypoint to write to disk.

    Modes:
      * default — probe only (unchanged behaviour).
      * ``ablation`` — probe, then ablate gender+region indicators in the SAME
        container (reuses the loaded model + fresh hidden states on disk).
      * ``ablation_only`` — skip probing; restore a prior run's probe, hidden
        states and dataset from ``<from_run>/`` on the volume, then ablate.
    """
    import shutil
    import sys

    sys.path.insert(0, _PIPELINE_DIR)

    from config import PipelineConfig  # bundled module
    from main import run_pipeline  # the shared pipeline body (single source of truth)

    out_root = Path("/tmp/ir_out")
    results_dir = out_root / "results"
    plots_dir = out_root / "plots"
    data_dir = out_root / "data"

    # Ablation is Gender-only (probe target is P(Male); reads labels["Gender"]).
    if (ablation or ablation_only) and "Gender" not in attributes:
        raise ValueError(
            f"Ablation requires the 'Gender' attribute to be probed; got attributes={attributes}."
        )

    if ablation_only:
        # Restore a prior run's results/ (probe + .npz) and data/ from the volume.
        if not from_run:
            raise ValueError("ablation_only requires from_run (the source run_name on the volume).")
        source = Path(OUTPUT_MOUNT) / from_run
        needed = [
            source / "results" / "best_probe_gender.joblib",
            source / "results" / "hidden_states_personas.npz",
            source / "data" / "dataset_personas.json",
        ]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "ablation_only cannot find required artifact(s) on volume "
                f"'{OUTPUT_VOLUME}': {missing}. The source run '{from_run}' must have been produced "
                "by a probe run that persists the .npz + data/ (i.e. after this feature was added)."
            )
        results_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "results", results_dir, dirs_exist_ok=True)
        shutil.copytree(source / "data", data_dir, dirs_exist_ok=True)
        print(f"[modal-run] ablation_only: restored probe/hidden/data from '{from_run}/' on volume")
    else:
        config = PipelineConfig()
        config.data.personas_file = _PERSONAS_REMOTE
        config.data.data_dir = str(data_dir)
        config.data.attributes = list(attributes)
        config.data.include_partial = include_partial
        config.data.samples_per_group = samples
        config.data.context_mode = context_mode

        config.model.model_name = model_id
        config.model.device_map = "cuda"          # single-GPU container
        config.model.torch_dtype = dtype
        config.model.max_seq_length = max_seq_length
        # HF_TOKEN comes from the huggingface-token secret; needed for gated repos.
        config.model.hf_token = os.environ.get("HF_TOKEN") or None

        config.probe.token_position = token_position
        config.results_dir = str(results_dir)
        config.plots_dir = str(plots_dir)

        run_pipeline(config, skip_extraction=False)

    if ablation or ablation_only:
        from aggregate_word_analysis import DEFAULT_CONTROL_WORDS, run_ablation

        print(f"[modal-run] Running indicator ablation (subset_per_class={ablation_subset or 'all'}) ...")
        run_ablation(
            dataset_path=str(data_dir / "dataset_personas.json"),
            run_dir=str(results_dir),
            plot_dir=str(plots_dir),
            model_name=model_id,
            device_map="cuda",
            dtype=dtype,
            mapping_paths=_ABLATION_MAPPINGS,
            control_words=DEFAULT_CONTROL_WORDS,
            subset_per_class=ablation_subset,
            batch_size=batch_size,
        )

    # Collect results/ + plots/ + data/ into a flat {relpath: bytes} mapping.
    # Everything is persisted to the volume; the large hidden_states .npz is
    # excluded from the CLIENT download by default (multi-GB intermediate) unless
    # MODAL_IR_KEEP_HIDDEN=1. The tiny dataset JSON is always shipped back.
    keep_hidden = os.environ.get("MODAL_IR_KEEP_HIDDEN", "") == "1"
    volume_blobs: dict[str, bytes] = {}
    for sub in ("results", "plots", "data"):
        base = out_root / sub
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(out_root).as_posix()
            volume_blobs[rel] = path.read_bytes()

    # Download set = volume set minus the big .npz (unless kept).
    download_blobs = {
        rel: data for rel, data in volume_blobs.items()
        if keep_hidden or Path(rel).name != "hidden_states_personas.npz"
    }

    # Persist EVERYTHING (incl. .npz + data/) to the output volume under
    # <run_name>/ so a detached run keeps its results and ablation_only can
    # later restore them.
    dest_root = Path(OUTPUT_MOUNT) / run_name
    if dest_root.exists():
        shutil.rmtree(dest_root)
    for rel, data in volume_blobs.items():
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    output_vol.commit()  # flush to the volume so `modal volume get` sees it
    print(f"[modal-run] Persisted {len(volume_blobs)} file(s) to volume "
          f"'{OUTPUT_VOLUME}' under '{run_name}/'")

    print(f"[modal-run] Returning {len(download_blobs)} artifact file(s): {sorted(download_blobs)}")
    return download_blobs


@app.function(
    gpu=GPU,
    volumes={HF_CACHE: hf_cache_vol, OUTPUT_MOUNT: output_vol},
    secrets=_hf_secrets(),
    timeout=TIMEOUT,
)
def run_sweep(
    model_id: str,
    dtype: str,
    max_seq_length: int,
    from_run: str,
    run_name: str,
    batch_size: int = 4,
    conditions: list[str] | None = None,
) -> dict[str, bytes]:
    """Per-condition, per-layer 5-fold-CV Gender-probe accuracy sweep.

    Restores a prior probe run's dataset (and, for the FULL condition, its
    cached hidden states) from ``<from_run>/`` on the volume, then re-extracts
    the NEUTRAL / ONLY_<TYPE> conditions and probes every layer. Produces the
    accuracy-vs-layer chart + CSV. Persists them to ``<run_name>/sweep/`` on the
    volume and returns them for local download.
    """
    import sys

    sys.path.insert(0, _PIPELINE_DIR)

    from aggregate_word_analysis import DEFAULT_CONTROL_WORDS
    from condition_layer_sweep import run_condition_sweep

    if not from_run:
        raise ValueError("run_sweep requires from_run (the source probe run_name on the volume).")

    source = Path(OUTPUT_MOUNT) / from_run
    dataset_path = source / "data" / "dataset_personas.json"
    full_hidden = source / "results" / "hidden_states_personas.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Sweep cannot find {dataset_path} on volume '{OUTPUT_VOLUME}'. The source run "
            f"'{from_run}' must persist data/ (produced after the ablation feature was added)."
        )
    if not full_hidden.exists():
        print(f"[modal-run] WARN: no cached FULL hidden states at {full_hidden}; FULL will be re-extracted.")
        full_hidden = None

    # Regenerated-neutral dataset bundled into the image (built via prepare_dataset
    # from neutral_personas/). Only the NEUTRAL_GENERATED condition reads it.
    neutral_dataset = Path(_PIPELINE_DIR) / "neutral_data" / "dataset_personas.json"
    if neutral_dataset.exists():
        print(f"[modal-run] NEUTRAL_GENERATED dataset available at {neutral_dataset}")
    else:
        print(f"[modal-run] NOTE: no regenerated-neutral dataset at {neutral_dataset}")
        neutral_dataset = None

    # Write outputs AND per-condition checkpoints directly under the volume so a
    # spot-preemption restart resumes instead of redoing completed conditions.
    out_dir = Path(OUTPUT_MOUNT) / run_name / "sweep"
    ckpt_dir = out_dir / "checkpoints"

    def _commit(_condition: str) -> None:
        output_vol.commit()  # flush this condition's checkpoint to the volume

    run_condition_sweep(
        dataset_path=str(dataset_path),
        out_dir=str(out_dir),
        model_name=model_id,
        device_map="cuda",
        dtype=dtype,
        mapping_paths=_ABLATION_MAPPINGS,
        control_words=DEFAULT_CONTROL_WORDS,
        full_hidden_path=str(full_hidden) if full_hidden else None,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_cb=_commit,
        conditions=conditions,
        neutral_dataset_path=str(neutral_dataset) if neutral_dataset else None,
    )

    blobs: dict[str, bytes] = {}
    for path in out_dir.rglob("*"):
        if path.is_file():
            blobs[f"sweep/{path.relative_to(out_dir).as_posix()}"] = path.read_bytes()

    output_vol.commit()
    print(f"[modal-run] Persisted {len(blobs)} sweep file(s) to '{OUTPUT_VOLUME}' under '{run_name}/sweep/'")
    return blobs


@app.local_entrypoint()
def sweep(
    model: str = "google/gemma-4-31B-it",
    from_run: str = "",
    dtype: str = "bfloat16",
    max_seq_length: int = 2048,
    batch_size: int = 4,
    out: str = "",
    conditions: str = "",
) -> None:
    """Run the per-condition layer-accuracy sweep for a prior probe run.

        MODAL_IR_GPU=A100-80GB modal run --detach \\
            experiments/internal_representation/modal_run.py::sweep \\
            --from-run results_modal_gemma4_31b_ablation --model google/gemma-4-31B-it

    ``--from-run`` names the source run on the volume (its dataset + FULL hidden
    states are reused). Results land under ``<out>/sweep/`` locally and on the
    volume; ``out`` defaults to ``from_run``. ``--model``/``--dtype`` MUST match
    the run that produced the source hidden states.

    ``--conditions`` is a comma-separated list of conditions to run, e.g.
    ``--conditions NEUTRAL,ONLY_MOVIE``. Omit to run all default conditions.
    """
    if not from_run:
        raise SystemExit("sweep requires --from-run <prior probe run_name on the volume>")
    run_name = out or from_run
    conditions_list = [c.strip() for c in conditions.split(",") if c.strip()] or None

    print(f"[modal-run] sweep model={model} gpu={GPU} from_run={from_run}")
    if conditions_list:
        print(f"[modal-run] conditions (explicit): {conditions_list}")
    else:
        print("[modal-run] conditions: all defaults")
    blobs = run_sweep.remote(
        model_id=model,
        dtype=dtype,
        max_seq_length=max_seq_length,
        from_run=from_run,
        run_name=run_name,
        batch_size=batch_size,
        conditions=conditions_list,
    )

    out_dir = _HERE / run_name
    for rel, data in blobs.items():
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"[modal-run] Wrote {len(blobs)} sweep file(s) to {out_dir}")
    print(
        "[modal-run] If a --detach run disconnected before this, fetch with:\n"
        f"    modal volume get {OUTPUT_VOLUME} {run_name}/sweep/ {out_dir}/sweep"
    )


@app.local_entrypoint()
def main(
    model: str = "google/gemma-4-31B-it",
    attributes: str = "Gender",
    samples: int = 0,
    dtype: str = "bfloat16",
    max_seq_length: int = 2048,
    token_position: str = "last",
    context_mode: str = "full",
    include_partial: bool = False,
    out: str = "results_modal",
    ablation: bool = False,
    ablation_subset: int = 0,
    ablation_only: bool = False,
    from_run: str = "",
    batch_size: int = 4,
) -> None:
    """Launch the remote probe and write artifacts under ``<out>/`` locally.

    ``attributes`` is a space- or comma-separated list, e.g. "Gender Region".
    ``samples`` is the cap PER (gender, region) group; ``0`` (the default) means
    "use every labelled history" (~3773). Pass e.g. ``--samples 40`` for a quick
    subsample test.

    Ablation (Gender-only): ``--ablation`` runs the probe then ablates gender +
    region indicators in one container. ``--ablation-only --from-run <name>``
    skips probing and reuses a prior run's probe/hidden/dataset from the volume.
    ``--ablation-subset N`` caps the ablation to N Female + N Male histories
    (0 = all, still balanced) — the main cost lever.
    """
    attr_list = [a for a in attributes.replace(",", " ").split() if a]
    samples_arg = samples if samples and samples > 0 else None

    # --- validation -------------------------------------------------------- #
    if ablation_only and not from_run:
        raise SystemExit("--ablation-only requires --from-run <prior run_name on the volume>")
    if (ablation or ablation_only) and "Gender" not in attr_list:
        raise SystemExit(
            f"--ablation / --ablation-only require 'Gender' in --attributes; got {attr_list}"
        )
    if (ablation or ablation_only) and len(attr_list) > 1:
        extra = [a for a in attr_list if a != "Gender"]
        print(f"[modal-run] WARN: ablation is Gender-only; ignoring extra attribute(s): {extra}")

    # Stable run name for the output volume subdir (must match at fetch time).
    run_name = out

    print(f"[modal-run] model={model} gpu={GPU} attributes={attr_list}")
    if ablation_only:
        print(f"[modal-run] MODE: ablation-only (reusing '{from_run}/' from volume '{OUTPUT_VOLUME}'; probing skipped)")
    else:
        scope = "ALL labelled histories" if samples_arg is None else f"SUBSAMPLE — {samples_arg} per (gender, region) group"
        print(f"[modal-run] sample scope: {scope}")
    if ablation or ablation_only:
        # ~25 gender + ~56 region indicators + 10 controls ≈ 91 phrases; ablation
        # re-runs the model on each matching (history × phrase) pair.
        per_class = ablation_subset if ablation_subset > 0 else "all(balanced)"
        print(
            f"[modal-run] ablation: subset_per_class={per_class}; up to ~2×per_class×~91 forward passes "
            f"(gender+region). A full run (~1715/1715 × ~91) is hundreds of thousands of passes on a large "
            f"model — use --ablation-subset to cap."
        )
    print(f"[modal-run] results persist to volume '{OUTPUT_VOLUME}' under '{run_name}/'")
    blobs = run_probe.remote(
        model_id=model,
        attributes=attr_list,
        samples=samples_arg,
        dtype=dtype,
        max_seq_length=max_seq_length,
        token_position=token_position,
        context_mode=context_mode,
        include_partial=include_partial,
        run_name=run_name,
        ablation=ablation,
        ablation_subset=ablation_subset,
        ablation_only=ablation_only,
        from_run=from_run,
        batch_size=batch_size,
    )

    out_dir = _HERE / out
    for rel, data in blobs.items():
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"[modal-run] Wrote {len(blobs)} file(s) to {out_dir}")
    print(
        "[modal-run] If a --detach run disconnected before this, fetch results with:\n"
        f"    modal volume get {OUTPUT_VOLUME} {run_name}/ {out_dir}"
    )
