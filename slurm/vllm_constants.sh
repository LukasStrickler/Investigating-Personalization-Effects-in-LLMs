#!/usr/bin/env bash
# Pinned vLLM container defaults (sourced by launch/pull/submit scripts).
# After a successful dev-queue smoke test, bump VLLM_IMAGE_TAG here if needed.
#
# Override at submit time without editing files:
#   VLLM_IMAGE_TAG=v0.8.5 sbatch --export=ALL,... slurm/bwunicluster.sbatch
#
# shellcheck shell=bash

VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-v0.8.5}"
# Pyxis/Enroot accepts docker:// or bare image refs depending on site config.
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:${VLLM_IMAGE_TAG}}"
