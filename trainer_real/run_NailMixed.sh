#!/bin/bash
# NAIL-Mixed: convex blend of NAIL-F (forward MC KL) and NAIL-R (importance-
# weighted reverse KL with a fresh auxiliary student token) on the SAME greedy
# student prefix. Matches small-cot's `loss = mixed`, `method_family = nail`:
#
#     L(theta) = (1 - BETA) * L_forward + BETA * L_reverse
#
# Defaults:
#   STUDENT_TEMP = 0.0  (greedy NAIL prefix, shared by both arms)
#   AUX_SAMPLE   = 1    (paper-faithful reverse arm; drawing a' ~ pi_theta
#                        keeps the reverse-KL estimator unbiased even though
#                        rollouts are greedy)
#   BETA         = 0.5  (equal forward/reverse mix; override at the CLI)
#   EXPERT_TEMP  = 32.0 (noisy expert; same default as run_NailR.sh /
#                        run_NailF.sh — set EXPERT_TEMP=1.0 for clean expert)
#
# Usage (from NAIL repo root):
#   bash trainer_real/run_NailMixed.sh                              # beta=0.5
#   BETA=0.3 bash trainer_real/run_NailMixed.sh                     # 70% F / 30% R
#   BETA=0.7 EXPERT_TEMP=1.0 bash trainer_real/run_NailMixed.sh     # clean expert, R-heavy
#
# All other knobs (GPU, SEED, BSZ, LR, …) follow the same env-var convention
# as run_reverse_lora.sh / run_forward_lora.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STUDENT=${STUDENT:-google/gemma-3-270m-it}
STUDENT_SHORT=${STUDENT_SHORT:-gemma270m_it}
EXPERT=${EXPERT:-google/gemma-3-1b-it}
EXPERT_SHORT=${EXPERT_SHORT:-gemma3_1b_it}
TRAIN_DATA=${TRAIN_DATA:-data/tinygsm/tinygsm_400k.jsonl}
PROMPT_FIELD=${PROMPT_FIELD:-question}
GPU=${GPU:-0}

EPOCHS=${EPOCHS:-1}
BSZ=${BSZ:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
STUDENT_TEMP=${STUDENT_TEMP:-0.0}
EXPERT_TEMP=${EXPERT_TEMP:-32.0}
BETA=${BETA:-0.5}
AUX_SAMPLE=${AUX_SAMPLE:-1}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-50}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-NAIL}

LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

GSM8K_EVAL_LOSS_DATA=${GSM8K_EVAL_LOSS_DATA:-data/gsm8k/test.jsonl}
GSM8K_EVAL_LOSS_BSZ=${GSM8K_EVAL_LOSS_BSZ:-8}

export WANDB_PROJECT

if [ "$STUDENT_TEMP" = "0" ] || [ "$STUDENT_TEMP" = "0.0" ]; then
    S_MODE="sgreedy"
else
    S_MODE="st$(echo $STUDENT_TEMP | tr '.' 'p')"
fi
if [ "$EXPERT_TEMP" = "0" ] || [ "$EXPERT_TEMP" = "0.0" ]; then
    E_MODE="egreedy"
else
    E_MODE="et$(echo $EXPERT_TEMP | tr '.' 'p')"
fi
BETA_TAG="beta$(echo $BETA | tr '.' 'p')"

RUN_NAME=${RUN_NAME:-mixed_lora_r${LORA_RANK}_${STUDENT_SHORT}_${EXPERT_SHORT}_${S_MODE}_${E_MODE}_${BETA_TAG}_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-output/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Train data not found: ${TRAIN_DATA}"
    exit 1
fi

LORA_ALPHA_FLAG=""
[ -n "$LORA_ALPHA" ] && LORA_ALPHA_FLAG="--lora_alpha $LORA_ALPHA"

AUX_SAMPLE_FLAG=""
case "$(echo "$AUX_SAMPLE" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) AUX_SAMPLE_FLAG="--aux_sample" ;;
esac

echo "=== NAIL-Mixed + LoRA (rank=${LORA_RANK}) ==="
echo "  Student:   ${STUDENT} (${S_MODE})"
echo "  Expert:    ${EXPERT} (${E_MODE})"
echo "  Beta:      ${BETA}  ((1-beta)*forward + beta*reverse)"
echo "  AuxSample: ${AUX_SAMPLE}  ${AUX_SAMPLE_FLAG:+(reverse arm uses fresh student sample)}"
echo "  Train:     ${TRAIN_DATA} (field=${PROMPT_FIELD})"
echo "  GPU:       ${GPU}"
echo "  Eff BSZ:   $((BSZ * GRAD_ACCUM)) (bsz=${BSZ} × ga=${GRAD_ACCUM})"
echo "  LR:        ${LR}, Epochs: ${EPOCHS}, MaxNewTok: ${MAX_NEW_TOKENS}"
echo "  Output:    ${OUTPUT_DIR}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU nohup python "$SCRIPT_DIR/mixed_lora.py" \
    --student_model "$STUDENT" \
    --expert_model "$EXPERT" \
    --train_data "$TRAIN_DATA" \
    --prompt_field "$PROMPT_FIELD" \
    --output_dir "$OUTPUT_DIR" \
    --name "$RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --num_train_epochs "$EPOCHS" \
    --batch_size "$BSZ" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 50 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --bf16 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --student_temperature "$STUDENT_TEMP" \
    --expert_temperature "$EXPERT_TEMP" \
    --beta "$BETA" \
    $AUX_SAMPLE_FLAG \
    --lora_rank "$LORA_RANK" \
    $LORA_ALPHA_FLAG \
    --lora_dropout "$LORA_DROPOUT" \
    --gsm8k_eval_loss_data "$GSM8K_EVAL_LOSS_DATA" \
    --gsm8k_eval_loss_batch_size "$GSM8K_EVAL_LOSS_BSZ" \
    --skip_initial_eval \
    > "${LOG_DIR}/${RUN_NAME}.log" 2>&1 &

echo "  PID: $!"
echo "  Monitor: tail -f ${LOG_DIR}/${RUN_NAME}.log"
