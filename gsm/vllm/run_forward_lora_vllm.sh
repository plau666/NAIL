#!/bin/bash
# Forward-KL on-policy distillation (NAIL-F / OPD-F) with LoRA: student NLL on the expert.s sampled token.
# Logs GSM8K test eval_loss every save_steps.
#
# Usage (from inside gsm/):
#   bash run_forward_lora.sh
#
# Override via env vars, e.g.:
#   GPU=3 STUDENT_TEMP=1.0 EXPERT_TEMP=1.0 bash run_forward_lora.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STUDENT=${STUDENT:-google/gemma-3-270m-it}
STUDENT_SHORT=${STUDENT_SHORT:-gemma270m_it}
EXPERT=${EXPERT:-google/gemma-3-1b-it}
EXPERT_SHORT=${EXPERT_SHORT:-gemma3_1b_it}
# Prompts-only file (expects `question` field by default; override with PROMPT_FIELD).
TRAIN_DATA=${TRAIN_DATA:-data/tinygsm/tinygsm_400k.jsonl}
PROMPT_FIELD=${PROMPT_FIELD:-question}
GPU=${GPU:-0}

EPOCHS=${EPOCHS:-1}
BSZ=${BSZ:-2}
GRAD_ACCUM=${GRAD_ACCUM:-32}
LR=${LR:-1e-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
STUDENT_TEMP=${STUDENT_TEMP:-1.0}
EXPERT_TEMP=${EXPERT_TEMP:-1.0}
SAVE_STEPS=${SAVE_STEPS:-200}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-NAIL}

LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

GSM8K_EVAL_LOSS_DATA=${GSM8K_EVAL_LOSS_DATA:-data/gsm8k/test.jsonl}
GSM8K_EVAL_LOSS_BSZ=${GSM8K_EVAL_LOSS_BSZ:-8}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.15}

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

RUN_NAME=${RUN_NAME:-forward_lora_r${LORA_RANK}_${STUDENT_SHORT}_${EXPERT_SHORT}_${S_MODE}_${E_MODE}_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-output/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Train data not found: ${TRAIN_DATA}"
    exit 1
fi

LORA_ALPHA_FLAG=""
[ -n "$LORA_ALPHA" ] && LORA_ALPHA_FLAG="--lora_alpha $LORA_ALPHA"

echo "=== Forward-KL + LoRA (rank=${LORA_RANK}) ==="
echo "  Student:   ${STUDENT} (${S_MODE})"
echo "  Expert:    ${EXPERT} (${E_MODE})"
echo "  Train:     ${TRAIN_DATA} (field=${PROMPT_FIELD})"
echo "  GPU:       ${GPU}"
echo "  Eff BSZ:   $((BSZ * GRAD_ACCUM)) (bsz=${BSZ} × ga=${GRAD_ACCUM})"
echo "  LR:        ${LR}, Epochs: ${EPOCHS}, MaxNewTok: ${MAX_NEW_TOKENS}"
echo "  Output:    ${OUTPUT_DIR}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python "$SCRIPT_DIR/forward_lora_vllm.py" \
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
    --max_grad_norm "$MAX_GRAD_NORM" \
    --logging_steps 50 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 50 \
    --bf16 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --student_temperature "$STUDENT_TEMP" \
    --expert_temperature "$EXPERT_TEMP" \
    --lora_rank "$LORA_RANK" \
    $LORA_ALPHA_FLAG \
    --lora_dropout "$LORA_DROPOUT" \
    --gsm8k_eval_loss_data "$GSM8K_EVAL_LOSS_DATA" \
    --gsm8k_eval_loss_batch_size "$GSM8K_EVAL_LOSS_BSZ" \
    --vllm_gpu_mem_util "$VLLM_GPU_MEM_UTIL" \
    --skip_initial_eval \
    > "${OUTPUT_DIR}/train.log" 2>&1 &

echo "  PID: $!"
echo "  Monitor: tail -f ${OUTPUT_DIR}/train.log"
