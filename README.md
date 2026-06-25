# NAIL — Noise-robust Aggregation for Imitation Learning

The repo has two experiment stacks:

| Directory | What it runs |
|---|---|
| [`gsm/`](gsm/README.md) | LoRA distillation on GSM8K/TinyGSM with Gemma student/expert models. |
| [`modadd/`](modadd/README.md) | Modular-addition experiments with a small transformer trained from scratch. |

## Setup

Install once from the repo root:

```bash
uv sync --locked
source .venv/bin/activate
```

This creates `.venv/` from `uv.lock` and installs the dependencies for both
experiment stacks, including PyTorch CUDA 12.8, vLLM, Transformers, PEFT,
Hydra, and W&B.

Expected core versions after `uv sync --locked`:

```text
vLLM:                0.20.2+cu129
torch:               2.11.0+cu128
transformers:        5.8.1
peft:                0.18.1
```

Use `uv sync --locked`, not an unlocked dependency solve. `pyproject.toml` pins
vLLM to the official CUDA-12.9 wheel artifact, and that wheel has been checked
to link against `libcudart.so.12`, matching the CUDA-12 runtime installed by the
PyTorch CUDA 12.8 wheels.

Then choose a stack:

```bash
cd gsm      # real-model GSM8K/TinyGSM experiments
# or
cd modadd   # modular-addition experiments
```

See [`gsm/README.md`](gsm/README.md) and [`modadd/README.md`](modadd/README.md)
for the full commands.

## Method Map

GSM commands below are run from inside `gsm/`.

| Method | GSM command | Modadd command |
|---|---|---|
| LogLossBC | `bash scripts/train.sh configs/offline_bc.yaml` | `python -m nanogpt.run experiment=modadd_noisy_bc` |
| NAIL-F | `bash scripts/train.sh configs/nail_f.yaml` | `python -m nanogpt.run experiment=modadd_nail` |
| NAIL-R | `bash scripts/train.sh configs/nail_r.yaml` | `python -m nanogpt.run experiment=modadd_nail_reverse_mc_fixed` |
| NAIL-Mixed | `bash scripts/train.sh configs/nail_mixed.yaml` | `python -m nanogpt.run experiment=modadd_nail task.loss=mixed task.kl_beta=<beta>` |
| OPD-F | `bash scripts/train.sh configs/opd_f.yaml` | `python -m nanogpt.run experiment=modadd_opd_forward` |
| OPD-R | `bash scripts/train.sh configs/opd_r.yaml` | `python -m nanogpt.run experiment=modadd_opd` |


## Attribution

The base causal transformer in `modadd/model.py` and the `nanogpt` package name
are derived from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT),
MIT licensed. 
