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
│   ├── forward_lora.py          # Simple-OPD training (custom loop)
│   ├── reverse_lora.py              # TM-OPD training (custom loop)
│   ├── run_offline_bc_lora.sh      # launcher for OBC
│   ├── run_forward_lora.sh      # launcher for Simple OPD
│   └── run_reverse_lora.sh          # launcher for TM-OPD
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

We pin every package (including torch / vllm with the right CUDA local-version
tags) in `requirements.txt`. The recommended way to install is with
[`uv`](https://docs.astral.sh/uv/) — it's much faster than `pip` and resolves
the torch + vllm CUDA wheels cleanly:

```bash
# 1. Install uv once (skip if you already have it).
curl -LsSf https://astral.sh/uv/install.sh | sh
#  (or:  pip install --user uv  if you'd rather not pipe curl)

# 2. Create a Python 3.11 venv inside the repo and install the frozen deps.
cd <your_repo_path>
uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements.txt
```

`torch==2.11.0+cu128` lives on the PyTorch index, hence the `--extra-index-url`.
`vllm==0.20.2+cu124` is pulled from PyPI; uv will build it from source if no
prebuilt wheel matches your platform — that's ~15–40 min on a 96-core box, but
only once. Subsequent venvs reuse the build cache.

Plain `pip + venv` still works (slower):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
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

### Forward KL distillation (on-policy, hard labels from expert argmax)

The base launcher is `trainer_real/run_forward_lora.sh`. Four named
wrappers cover the canonical (`student_temp`) × (`clean/noisy expert`)
combinations:

| Wrapper | Method | `STUDENT_TEMP` |
|---|---|---|
| `run_NailF.sh` | NAIL-F | 0 (greedy student rollouts) |
| `run_OpdF.sh`  | OPD-F  | 1 (sampled student rollouts) |

Each wrapper just `exec`s `run_forward_lora.sh` with `STUDENT_TEMP` locked
and `EXPERT_TEMP` / `GRAD_ACCUM` / `GPU` / `SEED` / `TRAIN_DATA` / … left
overridable. Toggle clean vs noisy expert via `EXPERT_TEMP`:

```bash
# NAIL-F  (greedy student rollouts)
EXPERT_TEMP=1.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_NailF.sh   # clean expert
EXPERT_TEMP=4.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_NailF.sh   # noisy expert

# OPD-F   (temp-1 student rollouts)
EXPERT_TEMP=1.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_OpdF.sh    # clean expert
EXPERT_TEMP=4.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_OpdF.sh    # noisy expert
```

If you'd rather hand-tune everything, call the base script directly:

```bash
STUDENT_TEMP=1.0 EXPERT_TEMP=1.0 GPU=3 bash trainer_real/run_forward_lora.sh
```

### Reverse KL distillation (on-policy, importance-weighted policy gradient)

Same shape as forward; the reverse-KL variant uses
`trainer_real/run_reverse_lora.sh`:

| Wrapper | Method | `STUDENT_TEMP` |
|---|---|---|
| `run_NailR.sh` | NAIL-R | 0 (greedy student rollouts) |
| `run_OpdR.sh`  | OPD-R  | 1 (sampled student rollouts) |

```bash
# NAIL-R  (greedy student rollouts)
EXPERT_TEMP=1.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_NailR.sh   # clean expert
EXPERT_TEMP=4.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_NailR.sh   # noisy expert

# OPD-R   (temp-1 student rollouts)
EXPERT_TEMP=1.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_OpdR.sh    # clean expert
EXPERT_TEMP=4.0 TRAIN_DATA=data/tinygsm/tinygsm_400k.jsonl bash trainer_real/run_OpdR.sh    # noisy expert
```

Or call the base script directly:

```bash
STUDENT_TEMP=1.0 EXPERT_TEMP=1.0 GPU=3 bash trainer_real/run_reverse_lora.sh
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
- **Gradient clipping**: disabled in all three methods. forward/reverse at high LR (≥ 5e-4) can diverge
- **LR schedule**: all three use linear warmup (10%) → cosine decay to 1% of peak LR.
- **bf16**: model weights & activations in bf16. OBC upcasts logits to fp32 for CE via Accelerate's autocast; forward/reverse keep loss in bf16.
- **Memory** on a 40 GiB A100:
  - OBC: bsz=8 ga=8 max_length=768 → ~37 GiB (tight)
  - forward / reverse at bsz=2 ga=32 rollout=64 → ~25 GiB
  - forward / reverse at bsz=2 ga=256 rollout=512 → ~25 GiB (best per-prompt throughput)
  - forward / reverse at bsz=2 ga=512 rollout=1024 → ~38 GiB (risky)
- **Per-step time** (forward on gemma-3-270m + 1B, mnt=512):
  - rollout=64: ~41 s/step (0.64 s/prompt)
  - rollout=512: ~95 s/step (0.19 s/prompt)
  - rollout=1024: ~164 s/step (0.16 s/prompt)

## Code reference

- Shared utilities: `trainer_real/gsm_utils.py` (dataset, collator, chat-template builder, GSM8K answer extractor, eval-loss helper).
- Chat format is always Gemma-3-IT: `<bos><start_of_turn>user ...<end_of_turn><start_of_turn>model ...<end_of_turn>`.
- System prompt (shared across all three methods + eval):
  `"Please reason step by step, and put your final answer within \\boxed{}."`
