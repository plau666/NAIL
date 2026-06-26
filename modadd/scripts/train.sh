#!/usr/bin/env bash
# Thin launcher for modular-addition experiments. The method presets live in
# hydra_configs/experiment/*.yaml; extra KEY=VALUE arguments are Hydra
# overrides passed through to nanogpt.run.
#
# Examples:
#   bash scripts/train.sh experiment=modadd_prompt_bank
#   bash scripts/train.sh experiment=modadd_cot
#   bash scripts/train.sh experiment=modadd_nail
#   bash scripts/train.sh experiment=modadd_nail task.loss=mixed task.kl_beta=0.25
#   bash scripts/train.sh experiment=modadd_opd optim.seed=43 task.eta=0.1
#   bash scripts/train.sh experiment=modadd_noisy_bc run.name=bc_eta005_seed44 optim.seed=44
#
# Common overrides:
#   runtime=cpu, runtime.device=cuda:1, logging=disabled, run.name, run.out_dir,
#   task.eta, task.subset_size, task.loss, task.kl_beta, optim.seed,
#   optim.batch_size, optim.max_iters, optim.learning_rate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODADD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$MODADD_DIR"

exec "${PYTHON:-python}" -m nanogpt.run "$@"
