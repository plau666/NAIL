#!/bin/bash
# NAIL-R (reverse) = run_reverse_lora.sh with student rollouts pinned to greedy
# (STUDENT_TEMP=0) and the aux-sample reverse-KL estimator
# (AUX_SAMPLE=1). All other knobs (GPU, EXPERT_TEMP, SEED, BSZ, …) are passed
# through; override them as you would for the base script.
#
# Usage (from inside gsm/):
#   bash run_NailR.sh
#
# Examples:
#     - Noisy expert:
#        EXPERT_TEMP=4.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash run_NailR.sh
#     - Clean expert:
#        EXPERT_TEMP=1.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash run_NailR.sh

#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_TEMP=${EXPERT_TEMP:-4.0} \
GRAD_ACCUM=${GRAD_ACCUM:-2} \
STUDENT_TEMP=0.0 \
AUX_SAMPLE=${AUX_SAMPLE:-1} \
    exec bash "$SCRIPT_DIR/run_reverse_lora.sh" "$@"
