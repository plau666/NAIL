# Modular-Addition Experiments

Modular-addition experiments for LogLossBC, NAIL-F, NAIL-R, NAIL-Mixed, OPD-F,
and OPD-R. The student is a small transformer trained from scratch with Hydra
configs.

## Quick Start

1. Set up the environment from the repo root.

```bash
uv sync --locked
source .venv/bin/activate
cd modadd
```

2. Prepare the clean prompt bank.

```bash
bash scripts/train.sh experiment=modadd_prompt_bank
```

3. Pretrain the clean CoT teacher.

```bash
bash scripts/train.sh experiment=modadd_cot
```

4. Run one method. See [Training](#training).

```bash
bash scripts/train.sh experiment=<experiment>
```

If you want to run LogLossBC, render noisy teacher rollouts first; see
[OfflineBC](#offlinebc). The online methods do not need this step.

## Training

All methods use the same launcher:

```bash
bash scripts/train.sh experiment=<experiment> KEY=VALUE ...
```

Available experiments:

| Method | Experiment |
|---|---|
| LogLossBC | `modadd_noisy_bc` |
| NAIL-F | `modadd_nail` |
| NAIL-R | `modadd_nail_reverse_mc_fixed` |
| NAIL-Mixed | `modadd_nail` with `task.loss=mixed task.kl_beta=<beta>` |
| OPD-F | `modadd_opd_forward` |
| OPD-R | `modadd_opd` |

Common overrides:

```bash
bash scripts/train.sh experiment=modadd_nail task.eta=0.1 optim.seed=43
bash scripts/train.sh experiment=modadd_nail task.loss=mixed task.kl_beta=0.25
bash scripts/train.sh experiment=modadd_opd runtime.device=cuda:1 optim.batch_size=64
bash scripts/train.sh experiment=modadd_noisy_bc run.name=bc_eta005_seed44 optim.seed=44
```

Useful override keys include `runtime=cpu`, `runtime.device`, `logging`,
`run.name`, `run.out_dir`, `task.eta`, `task.subset_size`, `task.loss`,
`task.kl_beta`, `optim.seed`, `optim.batch_size`, `optim.max_iters`, and
`optim.learning_rate`.

Checkpoints and run metadata are written to `run.out_dir`. By default, public
configs write prompt banks and rendered datasets under `data/`, and teacher,
online, and OfflineBC runs under `reruns/`. Set `run.out_dir=<path>` when you
want a simple explicit output path.

## OfflineBC

LogLossBC trains on a fixed noisy-teacher rollout dataset. Render it first:

```bash
bash scripts/train.sh experiment=modadd_noisy_render task.eta=0.05 task.subset_size=1000000
```

Then train:

```bash
bash scripts/train.sh experiment=modadd_noisy_bc task.eta=0.05 task.subset_size=1000000
```

Use the same `task.eta`, `task.subset_size`, `task.modadd_p`, `task.modadd_m`,
and seed settings for rendering and training so the dataset name resolves to the
same directory.

## Method Map

The paper-facing methods are presets over a shared student-prefix backend:

| Method | Prefix policy | Loss | Main knobs |
|---|---|---|---|
| NAIL-F | greedy student | forward KL MC | `task.loss=forward`, rollout temp `0` |
| NAIL-R | greedy student | reverse KL MC | `task.loss=reverse`, rollout temp `0` |
| NAIL-Mixed | greedy student | mixed KL | `task.loss=mixed`, `task.kl_beta=<beta>` |
| OPD-F | sampled student | forward KL MC | `task.rollout_temperature_override=1.0` |
| OPD-R | sampled student | reverse KL MC | `experiment=modadd_opd` |
| LogLossBC | fixed teacher rollouts | token log loss | `experiment=modadd_noisy_bc` |

## Layout

```text
modadd/
  data/modular_addition/       # task and prompt-bank utilities
  data/common/                  # shared prompt-bank, render, eval, and loss helpers
  hydra_configs/               # public method, task, runtime, and sweep configs
  nanogpt/                     # training package and Hydra entry point
  scripts/train.sh             # thin launcher around python -m nanogpt.run
  model.py                     # small nanoGPT-style transformer
  nanogpt_checkpoint.py        # checkpoint I/O
```

## Notes

- W&B logging is enabled in GPU experiment presets. Use `logging=disabled` for
  local dry runs.
- Hydra writes launcher logs under `hydra_outputs/` or `hydra_multirun/`.
