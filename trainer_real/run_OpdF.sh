# OPD-F (forward) = run_forward_lora.sh with student rollouts at temperature 1
# (STUDENT_TEMP=1.0). All other knobs (GPU, EXPERT_TEMP, SEED, BSZ, …) are
# passed through; override them as you would for the base script.
#
# Usage (from NAIL repo root):
#   bash trainer_real/run_OpdF.sh
#
# Examples:
#     - Noisy expert:
#        EXPERT_TEMP=4.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_OpdF.sh
#     - Clean expert:
#        EXPERT_TEMP=1.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_OpdF.sh

#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_TEMP=${EXPERT_TEMP:-32.0} \
GRAD_ACCUM=${GRAD_ACCUM:-2} \
STUDENT_TEMP=1.0 \
    exec bash "$SCRIPT_DIR/run_forward_lora.sh" "$@"
