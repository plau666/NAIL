# NAIL — synthetic experiments (modular addition)

The synthetic-experiment side of NAIL. Implements **NAIL-F**, **NAIL-R**,
**OPD-F**, **OPD-R**, **NAIL-Mixed**, and the offline **LogLossBC** baseline
on the modular-addition task with a small transformer student trained from
scratch.

The `gsm/` directory implements the same methods for LoRA-finetuned
LLMs on GSM8K. The math and surrogates match between the two; the synth side
trains from scratch with a Hydra-driven launcher, and the real side uses
argparse + bash wrappers.

## Layout

```
modadd/
├── README.md
├── nanogpt/                  # core package — methods, trainers, pipelines
│   ├── methods/student_prefix.py     # NAIL-F/R + OPD-F/R + Mixed loss code
│   ├── trainers/native_student_prefix.py   # the main on-policy training loop
│   ├── trainers/{nail,opd,pretrain}.py     # method-family entrypoints
│   ├── trainers/{configs,runtime,wandb}.py # config schemas, RNG, logging
│   ├── pipelines/modadd_data.py            # modular-addition data pipeline
│   ├── workers/                            # pretraining workers
│   ├── utils/                              # repo / hydra resolvers
│   └── run.py                              # `@hydra.main` Hydra entry point
├── hydra_configs/            # config tree (config.yaml + experiment/, task/, …)
├── data/
│   ├── modular_addition/     # task definition + prompt-bank generators
│   ├── synthetic/            # shared infrastructure (corruption, eval, …)
│   └── s5_cot/               # kept only as an import dependency of student_prefix.py
│                             # (CORRUPTIBLE_IDS + semantic-key noise constants).
├── model.py                  # tiny transformer (nanoGPT-style)
├── nanogpt_checkpoint.py     # checkpoint I/O
└── torch_dtypes.py
```

## Install

Install everything from the top-level repo via `uv` per
[`../README.md`](../README.md#quick-start). `hydra-core`,
`hydra-submitit-launcher`, and `omegaconf` are pinned in the main
`requirements.txt`, so nothing additional is needed for `modadd/`.

## Running a modular-addition experiment

All commands assume `cd NAIL/modadd`. Hydra finds `nanogpt/` and the
`data.*` / `model` / `nanogpt_checkpoint` top-level helpers automatically
because Python uses the cwd in `sys.path`.

### Pretrain the teacher (modadd CoT)

```bash
python -m nanogpt.run experiment=modadd_cot
```

### Train a NAIL-F or NAIL-R student from the saved teacher

```bash
# NAIL training (forward / reverse / mixed via experiment overrides)
python -m nanogpt.run experiment=modadd_nail

# OPD baselines
python -m nanogpt.run experiment=modadd_opd
python -m nanogpt.run experiment=modadd_opd_forward

# Offline noisy-BC baseline
python -m nanogpt.run experiment=modadd_noisy_bc
```

Common overrides:

```bash
# Run on GPU 1, seed 43, change eta (noise level) for the teacher
python -m nanogpt.run experiment=modadd_nail \
    runtime.device=cuda:1 runtime.seed=43 task.eta=0.1

# Switch the reverse-KL beta in the mixed objective
python -m nanogpt.run experiment=modadd_nail \
    task.loss=mixed task.kl_beta=0.3
```

See `hydra_configs/experiment/modadd_*.yaml` for the full list of preset
configs. The list:

```
modadd_base.yaml                # base modadd student pretraining
modadd_base_p7_m30.yaml         # base with p=7, m=30
modadd_cot.yaml                 # CoT teacher pretrain
modadd_cot_p7_m21.yaml          # p=7, m=21 CoT variant
modadd_cot_p7_m31.yaml          # p=7, m=31 CoT variant
modadd_nail.yaml                # NAIL-F (default)
modadd_nail_reverse_full.yaml   # NAIL-R, full-distribution variant
modadd_nail_reverse_mc_fixed.yaml   # NAIL-R, MC, fixed eta
modadd_noisy_bc.yaml            # offline LogLossBC on noisy teacher rollouts
modadd_noisy_render.yaml        # generate noisy-teacher rollout dataset
modadd_opd.yaml                 # OPD-R (default reverse on-policy distillation)
modadd_opd_forward.yaml         # OPD-F
modadd_prompt_bank.yaml         # build the clean modadd prompt bank
```

## Method ↔ knob map

The paper-facing names map onto a few canonical config fields. See
`nanogpt/methods/student_prefix.py:90-114` (`resolved_paper_method_name`) for
the resolver, but the short version:

| Method      | `task.method_family` | `task.loss` | `task.teacher_signal` | `task.rollout_temperature_override` |
|-------------|----------------------|-------------|------------------------|-------------------------------------|
| NAIL-F      | nail                 | forward     | mc                     | 0.0 (default)                       |
| NAIL-R      | nail                 | reverse     | mc                     | 0.0                                 |
| NAIL-Mixed  | nail                 | mixed       | mc                     | 0.0 (set `task.kl_beta`)            |
| OPD-F       | opd                  | forward     | mc                     | 1.0                                 |
| OPD-R       | opd                  | reverse     | mc                     | 1.0                                 |
| LogLossBC   | offline baseline     | n/a         | n/a                    | n/a                                 |

Hydra `experiment=modadd_*` presets already wire these correctly — the table
is just for direct field overrides.
