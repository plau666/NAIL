#!/usr/bin/env bash
# Thin launcher for GSM experiments. Defaults and method-specific settings live
# in ../configs/*.yaml; train.py converts them to trainer CLI arguments.
#
# Examples:
#   bash scripts/train.sh configs/nail_f.yaml
#   bash scripts/train.sh configs/nail_mixed.yaml RUN_NAME=nail_mixed_seed44 SEED=44
#   bash scripts/train.sh configs/nail_r.yaml BSZ=4 GRAD_ACCUM=16 MAX_NEW_TOKENS=1024
#   bash scripts/train.sh configs/opd_f.yaml EXPERT_TEMP=1.0 GPU=1
#   bash scripts/train.sh configs/offline_bc.yaml BSZ=4 GRAD_ACCUM=16 MAX_LENGTH=1536
#
# Common overrides:
#   RUN_NAME, OUTPUT_DIR, GPU, SEED, BSZ, GRAD_ACCUM, LR,
#   MAX_NEW_TOKENS, MAX_LENGTH, STUDENT_TEMP, EXPERT_TEMP, BETA,
#   VLLM_GPU_MEM_UTIL, RESUME_FROM_CHECKPOINT, WANDB_PROJECT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$GSM_DIR"

exec "${PYTHON:-python}" train.py "$@"
