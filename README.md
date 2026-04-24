# NAIL — Noise-Aware Imitation Learning (GSM8K)

Three LoRA-based distillation methods for a small student LLM (e.g. gemma-3-270m-it) from a larger expert (gemma-3-1b-it), evaluated on GSM8K:

- **Offline BC** — SFT on pre-generated expert rollouts (off-policy).
- **Simple OPD** — on-policy distillation with expert's argmax tokens as hard labels.
- **TM-OPD** — on-policy distillation with importance-weighted policy-gradient loss.

## Layout

```
NAIL/
├── README.md
├── requirements.txt
├── trainer_real/
│   ├── gsm_utils.py                # shared dataset, collator, eval, answer extraction
│   ├── offline_bc_lora.py          # OBC training (uses HF Trainer)
│   ├── simple_opd_lora.py          # Simple-OPD training (custom loop)
│   ├── TM_opd_lora.py              # TM-OPD training (custom loop)
│   ├── run_offline_bc_lora.sh      # launcher for OBC
│   ├── run_simple_opd_lora.sh      # launcher for Simple OPD
│   └── run_TM_opd_lora.sh          # launcher for TM-OPD
├── eval/
│   ├── eval_gsm8k.py           # eval any HF model on GSM8K with vLLM
│   └── eval_lora_ckpts_gsm8k.py # sweep LoRA ckpts in a run dir with vLLM
└── data/
    ├── rollout.py              # generate teacher rollouts with vLLM
    ├── download_gsm8k.py
    ├── download_tinygsm.py
    ├── execute_tinygsm.py      # run ground-truth python answers for tinygsm
    ├── gsm8k/
    │   ├── test.jsonl          # 1319 examples
    │   └── train.jsonl         # 7473 examples
    └── tinygsm/
        └── tinygsm_400k.jsonl  # 400k (question, answer) prompts for on-policy methods
```

## Setup

```bash
cd /home/peihanliu/NAIL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU: tested on A100 40GB. bf16 throughout. A single GPU is sufficient.

Optionally log in to W&B (`wandb login`) — runs log loss, LR, and GSM8K eval loss.

## 1. Prepare data

GSM8K test/train are included (small). The `tinygsm_400k.jsonl` prompt file is also included (~80 MB).

**To build tinygsm from scratch** (~15 min):
```bash
python data/download_tinygsm.py        # fetch full TinyGSM, filter short programs
python data/execute_tinygsm.py         # run Python programs to get ground-truth answers
# Optional: slice to 400k
head -n 400000 data/tinygsm/tinygsm_all.jsonl > data/tinygsm/tinygsm_400k.jsonl
```

## 2. Generate teacher rollouts (required only for OBC)

Simple-OPD and TM-OPD don't need pre-generated rollouts — they use the prompt file directly.
OBC needs a (prompt, completion) jsonl produced by running the expert:

```bash
mkdir -p data/teacher_rollouts
CUDA_VISIBLE_DEVICES=0 python data/rollout.py \
    --model google/gemma-3-1b-it \
    --data data/tinygsm/tinygsm_400k.jsonl \
    --prompt_field question --gt_answer_field answer \
    --output data/teacher_rollouts/train.jsonl \
    --temperature 1.0 --top_p 1.0 --n 1 --seed 42 \
    --max_new_tokens 512 \
    --gpu_memory_utilization 0.5
```

For sharding across multiple GPUs, run the script per-shard and concatenate the jsonls.

## 3. Training

Run all launchers from the NAIL repo root.

### Offline BC (off-policy SFT on teacher rollouts)
```bash
bash trainer_real/run_offline_bc_lora.sh
# Override: STUDENT=google/gemma-3-1b-it GPU=3 LR=5e-4 bash trainer_real/run_offline_bc_lora.sh
```

### Simple OPD (on-policy, hard labels from expert argmax)
```bash
bash trainer_real/run_simple_opd_lora.sh
# Override: STUDENT_TEMP=1.0 EXPERT_TEMP=1.0 GPU=3 bash trainer_real/run_simple_opd_lora.sh
# Student temp 0 = greedy rollouts; expert temp controls target sharpness.
```

### TM-OPD (on-policy, importance-weighted policy gradient)
```bash
bash trainer_real/run_TM_opd_lora.sh
# Override: STUDENT_TEMP=1.0 EXPERT_TEMP=1.0 GPU=3 bash trainer_real/run_TM_opd_lora.sh
```

All three write LoRA checkpoints to `output/<run_name>/checkpoint-NNN/` and the
final adapter to `output/<run_name>/final/`. Training config is saved at
`output/<run_name>/config.json`.

## 4. Eval

### Sweep all ckpts of a LoRA run
```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_lora_ckpts_gsm8k.py \
    --run_dir output/<run_name>/ \
    --eval_data data/gsm8k/test.jsonl \
    --max_new_tokens 512 \
    --gpu_memory_utilization 0.5
# Writes <run_name>/ckpt_eval_summary.json and per-ckpt eval_results.json
```

### Eval a single HF model
```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_gsm8k.py \
    --model google/gemma-3-1b-it \
    --split test --mode greedy \
    --name gemma3_1b_it_greedy \
    --max_new_tokens 512
```

## Notes

- **Seeds**: all three scripts honor `--seed`. LoRA init, DataLoader shuffle, and rollout sampling are seeded. CUDA kernel nondeterminism is not disabled — two runs with the same seed match to ~0.5 ppt.
- **Gradient clipping**: disabled in all three methods. SOPD/TMOPD at high LR (≥ 5e-4) can diverge — if you scale batch up, consider re-enabling clipping.
- **LR schedule**: all three use linear warmup (10%) → cosine decay to 1% of peak LR.
- **bf16**: model weights & activations in bf16. OBC upcasts logits to fp32 for CE via Accelerate's autocast; SOPD/TMOPD keep loss in bf16.
- **Memory** on a 40 GiB A100:
  - OBC: bsz=8 ga=8 max_length=768 → ~37 GiB (tight)
  - SOPD / TMOPD at bsz=2 ga=32 rollout=64 → ~25 GiB
  - SOPD / TMOPD at bsz=2 ga=256 rollout=512 → ~25 GiB (best per-prompt throughput)
  - SOPD / TMOPD at bsz=2 ga=512 rollout=1024 → ~38 GiB (risky)
- **Per-step time** (sopd on gemma-3-270m + 1B, mnt=512):
  - rollout=64: ~41 s/step (0.64 s/prompt)
  - rollout=512: ~95 s/step (0.19 s/prompt)
  - rollout=1024: ~164 s/step (0.16 s/prompt)

## Code reference

- Shared utilities: `trainer_real/gsm_utils.py` (dataset, collator, chat-template builder, GSM8K answer extractor, eval-loss helper).
- Chat format is always Gemma-3-IT: `<bos><start_of_turn>user ...<end_of_turn><start_of_turn>model ...<end_of_turn>`.
- System prompt (shared across all three methods + eval):
  `"Please reason step by step, and put your final answer within \\boxed{}."`
