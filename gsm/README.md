# NAIL — real experiments (LoRA distillation on GSM8K)

LoRA-based distillation methods for a small student LLM (e.g. `gemma-3-270m-it`)
from a larger expert (e.g. `gemma-3-1b-it`), evaluated on GSM8K:

- **LogLossBC** (`offline_bc_lora.py`) — SFT on pre-generated noisy expert rollouts (off-policy baseline).
- **NAIL-F** / **OPD-F** (`forward_lora.py`) — on-policy forward-KL distillation with the expert's sampled token as the MC target.
- **NAIL-R** / **OPD-R** (`reverse_lora.py`) — on-policy reverse-KL distillation with the importance-weighted policy-gradient surrogate.
- **NAIL-Mixed** (`mixed_lora.py`) — convex blend `(1-β)·L_F + β·L_R` of the NAIL-F and NAIL-R losses on a shared greedy prefix.

The synthetic-experiment side (modular addition, from-scratch transformer) lives in [`../modadd/`](../modadd/README.md).

## Layout

```
gsm/
├── README.md
├── gsm_utils.py                # shared dataset, collator, eval, answer extraction
├── offline_bc_lora.py          # LogLossBC (uses HF Trainer)
├── forward_lora.py             # NAIL-F / OPD-F (custom loop)
├── reverse_lora.py             # NAIL-R / OPD-R (custom loop, supports --aux_sample)
├── mixed_lora.py               # NAIL-Mixed (custom loop)
├── run_offline_bc_lora.sh      # base launcher for LogLossBC
├── run_forward_lora.sh         # base launcher for forward-KL family
├── run_reverse_lora.sh         # base launcher for reverse-KL family
├── run_NailF.sh / run_OpdF.sh  # thin wrappers — STUDENT_TEMP pinned
├── run_NailR.sh / run_OpdR.sh  # thin wrappers — STUDENT_TEMP pinned, AUX_SAMPLE preset
├── run_NailMixed.sh            # NAIL-Mixed wrapper with BETA knob
├── eval/
│   ├── eval_gsm8k.py           # eval any HF model on GSM8K with vLLM
│   └── eval_lora_ckpts_gsm8k.py # sweep LoRA ckpts in a run dir with vLLM
└── data/
    ├── rollout.py              # generate teacher rollouts with vLLM (for LogLossBC)
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

Install everything from the top-level repo via `uv` per
[`../README.md`](../README.md#quick-start). One install command covers both
trainers; nothing additional is needed for `gsm/`.

GPU: tested on A100 40GB. bf16 throughout. A single GPU is sufficient.
Optionally `wandb login` — runs log loss, LR, and GSM8K eval loss to the
`NAIL` wandb project.

All commands below assume `cd gsm` from the repo root.

## 1. Prepare data

GSM8K test/train are included (small). The TinyGSM prompts are committed as
gzipped files (`data/tinygsm/tinygsm_400k.jsonl.gz`, ~25 MB compressed / ~80 MB
expanded). Unpack once after a fresh clone:

```bash
gunzip -k data/tinygsm/*.jsonl.gz
```

To rebuild tinygsm from scratch (~15 min):

```bash
python data/download_tinygsm.py        # fetch full TinyGSM, filter short programs
                                       #   → data/tinygsm/train_short.jsonl
python data/execute_tinygsm.py         # run Python programs to get ground-truth answers
                                       #   → data/tinygsm/train_short_with_answer.jsonl
head -n 400000 data/tinygsm/train_short_with_answer.jsonl > data/tinygsm/tinygsm_400k.jsonl
```

## 2. Generate teacher rollouts (required only for LogLossBC)

The on-policy methods (NAIL-F/R, OPD-F/R, NAIL-Mixed) don't need pre-generated
rollouts — they query the expert live. LogLossBC needs a `(prompt, completion)`
jsonl from running the expert:

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

For sharding across GPUs, run per-shard and concatenate the jsonls.

## 3. Training

### LogLossBC (off-policy SFT on teacher rollouts)

```bash
bash run_offline_bc_lora.sh
# Override: STUDENT=google/gemma-3-1b-it GPU=3 LR=5e-4 bash run_offline_bc_lora.sh
```

### Forward-KL distillation (on-policy, MC teacher target)

Base launcher is `run_forward_lora.sh`. Two thin wrappers cover the canonical
`student_temp` × `clean/noisy expert` combinations:

| Wrapper        | Method | `STUDENT_TEMP` |
|---|---|---|
| `run_NailF.sh` | NAIL-F | 0 (greedy student rollouts) |
| `run_OpdF.sh`  | OPD-F  | 1 (temp-1 sampled student rollouts) |

Each wrapper `exec`s `run_forward_lora.sh` with `STUDENT_TEMP` locked, leaving
`EXPERT_TEMP` / `GRAD_ACCUM` / `GPU` / `SEED` / `TRAIN_DATA` / … overridable.
Toggle clean vs noisy expert via `EXPERT_TEMP`:

```bash
# NAIL-F  (greedy student rollouts)
EXPERT_TEMP=1.0 bash run_NailF.sh   # clean expert
EXPERT_TEMP=4.0 bash run_NailF.sh   # noisy expert

# OPD-F   (temp-1 student rollouts)
EXPERT_TEMP=1.0 bash run_OpdF.sh    # clean expert
EXPERT_TEMP=4.0 bash run_OpdF.sh    # noisy expert
```

To hand-tune everything, call the base script directly:

```bash
STUDENT_TEMP=1.0 EXPERT_TEMP=1.0 GPU=3 bash run_forward_lora.sh
```

### Reverse-KL distillation (on-policy, importance-weighted policy gradient)

Same shape as forward; the reverse-KL variant uses `run_reverse_lora.sh`:

| Wrapper        | Method | `STUDENT_TEMP` | `AUX_SAMPLE` default |
|---|---|---|---|
| `run_NailR.sh` | NAIL-R | 0 (greedy) | **1** — paper-faithful, draws `ŷ_t ~ π_θ` at each prefix |
| `run_OpdR.sh`  | OPD-R  | 1 (temp-1) | **0** — rollout token reused as the MC sample |

```bash
# NAIL-R  (greedy student rollouts, paper-faithful aux sample)
EXPERT_TEMP=1.0 bash run_NailR.sh   # clean expert
EXPERT_TEMP=4.0 bash run_NailR.sh   # noisy expert

# OPD-R   (temp-1 student rollouts)
EXPERT_TEMP=1.0 bash run_OpdR.sh    # clean expert
EXPERT_TEMP=4.0 bash run_OpdR.sh    # noisy expert
```

Why the asymmetric `AUX_SAMPLE` default: NAIL-R rolls out greedily, so the
rollout token is the student's **argmax**, not a sample from the temp-1
distribution being optimized — reusing it as the reverse-KL MC sample is
biased. NAIL-R therefore defaults to drawing a fresh `ŷ_t ~ π_θ(·|s_t)` via
`--aux_sample`. OPD-R rolls out at temp 1, so the rollout token already IS a
draw from `π_θ`, and reusing it is unbiased.

Or call the base script directly:

```bash
STUDENT_TEMP=1.0 EXPERT_TEMP=1.0 GPU=3 bash run_reverse_lora.sh
```

### NAIL-Mixed (convex blend of NAIL-F and NAIL-R)

```bash
# β = 0.5 (default), noisy expert (EXPERT_TEMP=32.0 default)
bash run_NailMixed.sh

# β = 0.1, clean expert, seed 43
BETA=0.1 EXPERT_TEMP=1.0 SEED=43 bash run_NailMixed.sh
```

The loss is `(1-β)·L_NAIL-F + β·L_NAIL-R` on a shared greedy student prefix.
Both arms re-use the same student/expert forward passes; the reverse arm uses
the same `--aux_sample` path as NAIL-R (default on).

### Output paths

All training writes LoRA checkpoints to `output/<run_name>/checkpoint-NNN/`
and the final adapter to `output/<run_name>/final/`. Training config is saved
at `output/<run_name>/config.json`.

## 4. Eval

### Sweep all checkpoints of a LoRA run

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

- **Seeds**: all scripts honor `--seed`. LoRA init, DataLoader shuffle, and
  rollout sampling are seeded. CUDA kernel nondeterminism is not disabled —
  two runs with the same seed match to ~0.5 ppt.
- **Gradient clipping**: disabled in all methods. forward/reverse at high LR
  (≥ 5e-4) can diverge.
- **LR schedule**: linear warmup (10%) → cosine decay to 1% of peak LR.
- **bf16**: model weights & activations in bf16. LogLossBC upcasts logits to
  fp32 for CE via Accelerate's autocast; forward/reverse/mixed keep loss in
  bf16.
- **Memory** on a 40 GiB A100:
  - LogLossBC: bsz=8 ga=8 max_length=768 → ~37 GiB (tight)
  - forward / reverse at bsz=2 ga=32 rollout=64 → ~25 GiB
  - forward / reverse at bsz=2 ga=256 rollout=512 → ~25 GiB (best per-prompt throughput)
  - forward / reverse at bsz=2 ga=512 rollout=1024 → ~38 GiB (risky)
- **Per-step time** (forward on gemma-3-270m + 1B, mnt=512):
  - rollout=64: ~41 s/step (0.64 s/prompt)
  - rollout=512: ~95 s/step (0.19 s/prompt)
  - rollout=1024: ~164 s/step (0.16 s/prompt)

## Code reference

- Shared utilities: `gsm_utils.py` (dataset, collator, chat-template builder,
  GSM8K answer extractor, eval-loss helper).
- Chat format is always Gemma-3-IT:
  `<bos><start_of_turn>user ...<end_of_turn><start_of_turn>model ...<end_of_turn>`.
- System prompt (shared across all methods + eval):
  `"Please reason step by step, and put your final answer within \\boxed{}."`
