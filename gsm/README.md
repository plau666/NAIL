# GSM Experiments

LoRA distillation on TinyGSM prompts, evaluated on GSM8K. The default setup
distills `google/gemma-3-270m-it` from `google/gemma-3-1b-it`.

## Quick Start

1. Set up the environment from the repo root.

```bash
uv sync --locked
source .venv/bin/activate
cd gsm
```

2. Prepare TinyGSM prompts.

```bash
gunzip -k data/tinygsm/*.jsonl.gz
```

GSM8K train/test are already included in `data/gsm8k/`. TinyGSM prompts are
stored as `data/tinygsm/tinygsm_400k.jsonl.gz`; the `gunzip -k` command expands
them without deleting the compressed copy.

If you want to run LogLossBC, generate teacher rollouts first; see
[OfflineBC](#offlinebc). The online methods do not need this step.

3. Run one experiment. See [Training](#training).

```bash
bash scripts/train.sh configs/<config>.yaml
```

4. Evaluate a run. See [Evaluation](#evaluation).

```bash
python eval/eval.py --run_dir output/<run_name> --mnts 512
```

## Training

All methods use the same launcher:

```bash
bash scripts/train.sh <config.yaml> KEY=VALUE ...
```

Available configs:

| Method | Config |
|---|---|
| LogLossBC | `configs/offline_bc.yaml` |
| NAIL-F | `configs/nail_f.yaml` |
| NAIL-R | `configs/nail_r.yaml` |
| NAIL-Mixed | `configs/nail_mixed.yaml` |
| OPD-F | `configs/opd_f.yaml` |
| OPD-R | `configs/opd_r.yaml` |

Common overrides:

```bash
bash scripts/train.sh configs/nail_f.yaml EXPERT_TEMP=4.0 GPU=0
bash scripts/train.sh configs/nail_r.yaml RUN_NAME=nail_r_seed43 SEED=43
bash scripts/train.sh configs/nail_mixed.yaml BETA=0.25 BSZ=4 GRAD_ACCUM=16 MAX_NEW_TOKENS=1024
```

Useful override keys include `RUN_NAME`, `OUTPUT_DIR`, `GPU`, `SEED`, `BSZ`,
`GRAD_ACCUM`, `LR`, `MAX_NEW_TOKENS`, `MAX_LENGTH`, `STUDENT_TEMP`,
`EXPERT_TEMP`, `BETA`, `VLLM_GPU_MEM_UTIL`, `WANDB_PROJECT`, and
`RESUME_FROM_CHECKPOINT`.

Checkpoints are written to:

```text
output/<run_name>/checkpoint-*/
output/<run_name>/final/
output/<run_name>/config.json
```

## OfflineBC

The online methods above train directly from TinyGSM prompts. LogLossBC first
needs teacher completions from the expert:

```bash
mkdir -p data/teacher_rollouts
CUDA_VISIBLE_DEVICES=0 python data/teacher_rollout.py \
    --model google/gemma-3-1b-it \
    --data data/tinygsm/tinygsm_400k.jsonl \
    --prompt_field question --gt_answer_field answer \
    --output data/teacher_rollouts/train.jsonl \
    --temperature 1.0 --top_p 1.0 --n 1 --seed 42 \
    --max_new_tokens 1024 \
    --gpu_memory_utilization 0.5
```

Then train:

```bash
bash scripts/train.sh configs/offline_bc.yaml
```

## Evaluation

Use one evaluator for raw models, one adapter, or all checkpoints in a run.

Raw model:

```bash
python eval/eval.py \
    --model google/gemma-3-1b-it \
    --name gemma3_1b_it_greedy \
    --mnts 512
```

Single LoRA adapter:

```bash
python eval/eval.py \
    --model google/gemma-3-270m-it \
    --adapter output/<run_name>/checkpoint-200 \
    --name <run_name>_step200 \
    --mnts 512
```

All checkpoints in a run:

```bash
python eval/eval.py \
    --run_dir output/<run_name> \
    --mnts 512
```

Raw/full-model eval supports `--mode greedy`, `--mode pass@k`, and
`--mode maj@k`. LoRA eval is greedy-only. Raw-model and single-adapter evals
write under `eval/<name>/`; run sweeps write under `output/<run_name>/eval/`.
Pass multiple values to `--mnts`, such as `--mnts 512 1024`, to evaluate more
than one generation budget.

## Layout

```text
gsm/
  configs/              # public training configs
  data/                 # committed GSM8K data, TinyGSM prompts, rollout utility
  eval/                 # GSM8K evaluator
  scripts/train.sh      # thin launcher around train.py
  trainers/             # method implementations
  train.py              # config-driven training launcher
  gsm_utils.py          # shared training utilities
```

## Notes

- W&B logging uses `WANDB_PROJECT=NAIL` by default. Run `wandb login` first if
  you want online logging.
- On-policy methods use vLLM during training. Adjust `VLLM_GPU_MEM_UTIL` if the
  rollout engine needs more or less GPU memory.
