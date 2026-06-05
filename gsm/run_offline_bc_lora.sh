#!/bin/bash
# Offline BC with LoRA: SFT a student model on pre-generated teacher rollouts.
# Logs GSM8K test eval_loss every save_steps.
#
# Usage (from inside gsm/):
#   bash run_offline_bc_lora.sh
#
# Override defaults via env vars, e.g.:
#   STUDENT=google/gemma-3-270m-it GPU=3 bash run_offline_bc_lora.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STUDENT=${STUDENT:-google/gemma-3-270m-it}
STUDENT_SHORT=${STUDENT_SHORT:-gemma270m_it}
# Point at the teacher-rollout jsonl (expects fields `prompt` and `completion`).
# Generate with: data/rollout.py (see README).
TRAIN_DATA=${TRAIN_DATA:-data/teacher_rollouts/train.jsonl}
GPU=${GPU:-0}

EPOCHS=${EPOCHS:-1}
BSZ=${BSZ:-8}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-1e-4}
MAX_LENGTH=${MAX_LENGTH:-768}
SAVE_STEPS=${SAVE_STEPS:-200}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-NAIL}

LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

# GSM8K eval-loss (logged every SAVE_STEPS)
GSM8K_EVAL_LOSS_DATA=${GSM8K_EVAL_LOSS_DATA:-data/gsm8k/test.jsonl}
GSM8K_EVAL_LOSS_BSZ=${GSM8K_EVAL_LOSS_BSZ:-8}

export WANDB_PROJECT

RUN_NAME=${RUN_NAME:-obc_lora_r${LORA_RANK}_${STUDENT_SHORT}_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-output/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Train data not found: ${TRAIN_DATA}"
    echo "Generate with: CUDA_VISIBLE_DEVICES=0 python data/rollout.py ..."
    exit 1
fi

LORA_ALPHA_FLAG=""
[ -n "$LORA_ALPHA" ] && LORA_ALPHA_FLAG="--lora_alpha $LORA_ALPHA"

echo "=== Offline BC + LoRA (rank=${LORA_RANK}) ==="
echo "  Student:   ${STUDENT}"
echo "  Train:     ${TRAIN_DATA}"
echo "  GPU:       ${GPU}"
echo "  Eff BSZ:   $((BSZ * GRAD_ACCUM)) (bsz=${BSZ} × ga=${GRAD_ACCUM})"
echo "  LR:        ${LR}, Epochs: ${EPOCHS}, MaxLen: ${MAX_LENGTH}"
echo "  Output:    ${OUTPUT_DIR}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU nohup python "$SCRIPT_DIR/offline_bc_lora.py" \
    --student_model "$STUDENT" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --name "$RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BSZ" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 50 \
    --bf16 \
    --max_length "$MAX_LENGTH" \
    --lora_rank "$LORA_RANK" \
    $LORA_ALPHA_FLAG \
    --lora_dropout "$LORA_DROPOUT" \
    --gsm8k_eval_loss_data "$GSM8K_EVAL_LOSS_DATA" \
    --gsm8k_eval_loss_batch_size "$GSM8K_EVAL_LOSS_BSZ" \
    --skip_initial_eval \
    > "${LOG_DIR}/${RUN_NAME}.log" 2>&1 &

echo "  PID: $!"
echo "  Monitor: tail -f ${LOG_DIR}/${RUN_NAME}.log"
