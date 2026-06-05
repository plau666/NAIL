# NAIL — Noise-Aware Imitation Learning

Reference implementation for the methods in *Noise-Aware Imitation Learning*:
**NAIL-F**, **NAIL-R**, **NAIL-Mixed**, plus the **OPD-F** / **OPD-R** and
**LogLossBC** baselines.

Two self-contained training stacks live side by side:

| Directory | Setting | Student / data |
|---|---|---|
| [`gsm/`](gsm/README.md) | LoRA fine-tuning on GSM8K | `gemma-3-270m-it` student distilled from `gemma-3-1b-it` expert; argparse + bash launchers |
| [`modadd/`](modadd/README.md) | From-scratch on modular addition | Small transformer trained from scratch; Hydra-driven launcher |

Each directory has its own `README.md` with a complete install / data-gen /
train / eval recipe. They share no Python code — the synth side uses a
Hydra-driven nanoGPT-style trainer; the real side uses argparse + custom
training loops over Hugging Face + PEFT — but they implement the same
algorithmic objectives.

## Quick start

```bash
# clone, then install everything (gsm + modadd) once at the repo root
cd <repo_root>
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements.txt
```

`requirements.txt` covers both trainers — `torch`, `vllm`, `peft`,
`transformers`, `hydra-core`, `omegaconf`, and the pinned transitives. After
that, `cd gsm` or `cd modadd` and follow the sub-README.

## Method ↔ entry-point map

| Paper method | Real (LoRA / GSM8K) | Synth (Hydra / modadd) |
|---|---|---|
| LogLossBC   | `gsm/run_offline_bc_lora.sh` | `python -m nanogpt.run experiment=modadd_noisy_bc` |
| NAIL-F      | `gsm/run_NailF.sh`           | `python -m nanogpt.run experiment=modadd_nail` |
| NAIL-R      | `gsm/run_NailR.sh`           | `python -m nanogpt.run experiment=modadd_nail_reverse_mc_fixed` |
| NAIL-Mixed  | `gsm/run_NailMixed.sh`       | `python -m nanogpt.run experiment=modadd_nail task.loss=mixed task.kl_beta=…` |
| OPD-F       | `gsm/run_OpdF.sh`            | `python -m nanogpt.run experiment=modadd_opd_forward` |
| OPD-R       | `gsm/run_OpdR.sh`            | `python -m nanogpt.run experiment=modadd_opd` |

## Hardware

All experiments run on a single A100 40 GB. bf16 throughout. Multi-GPU is not
required for any result.

## Attribution

The base causal transformer in `modadd/model.py` and the `nanogpt`
package name are derived from Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), MIT licensed (see `LICENSE`).
All training-loop code under `modadd/nanogpt/{methods,trainers,pipelines,workers}/`
and all of `gsm/` is original to NAIL.
