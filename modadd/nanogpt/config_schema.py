from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, MISSING, OmegaConf


@dataclass
class PipelineConfig:
    name: str = MISSING


@dataclass
class RunConfig:
    output_root: str = "."
    name: str = MISSING
    out_dir: str = MISSING


@dataclass
class LoggingConfig:
    wandb_log: bool = False
    wandb_project: str = "NAIL"
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_init_timeout: int = 300


@dataclass
class TorchrunConfig:
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    standalone: bool = True


@dataclass
class RuntimeConfig:
    python_bin: str = ""
    device: str = "cuda"
    dtype: str = "float16"
    compile: bool = False
    backend: str = "nccl"
    torchrun: TorchrunConfig = field(default_factory=TorchrunConfig)


@dataclass
class ClusterConfig:
    name: str = "local"
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem_gb: int = 32
    timeout_min: int = 720


@dataclass
class SweepConfig:
    name: str = "none"


@dataclass
class RandomSuffixNoiseConfig:
    enabled: bool = True
    key_positions: str = "semantic_key"
    trigger_eta: Optional[float] = None
    random_suffix_mode: str = "valid_tokens"
    keep_format_tokens: bool = True
    seed: int = 1337
    apply_to: str = "modadd"
    coord_strategy: str = "cyclic"
    fixed_coord: int = 0
    eligible_values: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    one_key_per_block: bool = True


@dataclass
class TaskConfig:
    """Hydra task surface for modular-addition experiments.

    Paper-to-code map:
    - `eta` is the paper noise level.
    - `teacher_checkpoint` is the frozen clean expert.
    - `teacher_law` selects the noisy expert construction.
    - `loss`, `teacher_signal`, and rollout/loss temperatures select the
      paper method preset; the online implementation backend is student_prefix.
    """

    dataset: str = "modadd_cot"
    dataset_prefix: str = ""
    run_prefix: str = ""
    out_prefix: str = ""
    data_root: str = "data"
    teacher_output_root: str = "."
    task: str = "modadd"
    modadd_p: int = 7
    modadd_m: int = 21
    bank_seed: int = 1337
    teacher_seed: int = 1337
    render_seed: int = 1337
    teacher_depth: int = 1
    gen_batch_size: int = 1024
    n_train: int = 0
    n_val: int = 0
    subset_size: int = 0
    # Paper eta: corruption/mixing rate for the selected noisy teacher law.
    eta: float = 0.0
    # Clean expert checkpoint pi*: queried directly online, or rolled out once
    # to render fixed LogLossBC datasets.
    teacher_checkpoint: str = ""
    prompt_bank_dir: str = ""
    # Noisy teacher law pi_eta. Offline render jobs store the same law in
    # `meta.json` so LogLossBC runs can be matched to student-prefix runs.
    teacher_law: str = "distributional_noise"
    # `teacher_signal=mc` samples teacher tokens; `full` uses the full
    # next-token teacher distribution when the law supports it.
    teacher_signal: str = "mc"
    # `forward`, `reverse`, `mixed`, and `jsd` are the code-level switches for
    # the KL directions and ablations described in the paper appendix.
    loss: str = "reverse"
    # For `mixed` and `jsd`, beta is the reverse/student-side weight in code:
    # beta=0 is forward-only, beta=1 is reverse-only.
    kl_beta: Optional[float] = None
    # Rollout temperature controls which prefixes are visited. It does not
    # change the default temperature-one student distribution used in the loss.
    rollout_temperature_override: Optional[float] = None
    # Optional loss-side temperature for forward/mixed/JSD ablations.
    loss_temperature_override: Optional[float] = None
    # ------------------------------------------------------------------
    # Legacy checkpoint/config aliases. Do not use in new configs.
    # Old checkpoints and analysis metadata may contain these fields; native
    # launches should use the explicit canonical knobs above.
    # ------------------------------------------------------------------
    objective: Optional[str] = None
    student_temperature: Optional[float] = None
    student_rollout_temperature: Optional[float] = None
    # Offline rendering controls for fixed LogLossBC datasets.
    rollout_mode: str = "greedy_then_corrupt"
    target_mode: str = "tokens"
    random_suffix_noise: RandomSuffixNoiseConfig = field(default_factory=RandomSuffixNoiseConfig)


@dataclass
class ModelConfig:
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    dropout: Optional[float] = None
    bias: Optional[bool] = None
    block_size: Optional[int] = None


@dataclass
class OptimConfig:
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"
    init_from_ckpt: str = ""
    continue_from_subset_size: int = 0
    gradient_accumulation_steps: int = 40
    batch_size: int = 12
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    save_every: int = 0
    save_interval: int = 0
    single_epoch: bool = False
    shuffle_prompts: bool = False
    seed: int = 1337
    eval_n: int = 5000
    eval_batch_size: int = 256
    eps: float = 1e-10
    offline_single_epoch: bool = False
    offline_eval_full: bool = True
    offline_train_subset_size: int = 0
    offline_train_shuffle: bool = False
    offline_target_type: str = "tokens"
    final_eval_on_exit: bool = False
    modadd_eval_metrics: bool = False
    modadd_eval_clean_train_loss: bool = False
    eval_seed: int = 123


@dataclass
class AppConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    run: RunConfig = field(default_factory=RunConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def materialize_config(raw_cfg: DictConfig) -> AppConfig:
    structured = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(structured, raw_cfg)
    cfg = OmegaConf.to_object(merged)
    assert isinstance(cfg, AppConfig)

    if not cfg.runtime.python_bin:
        cfg.runtime.python_bin = sys.executable
    if not cfg.logging.wandb_run_name:
        cfg.logging.wandb_run_name = cfg.run.name
    if not cfg.run.out_dir:
        cfg.run.out_dir = str(Path(cfg.run.output_root) / cfg.run.name)

    if cfg.pipeline.name not in {
        "pretrain",
        "opd",
        "nail",
        "student_prefix",
        "modadd_prompt_bank",
        "modadd_render",
    }:
        raise ValueError(f"unsupported pipeline {cfg.pipeline.name!r}")
    if cfg.pipeline.name == "pretrain" and not cfg.task.dataset:
        raise ValueError("pretrain experiments require task.dataset")
    # `student_prefix` is the neutral online backend name. `nail` is a legacy
    # alias for the same backend and remains supported for old configs.
    if cfg.pipeline.name in {"opd", "nail", "student_prefix"}:
        if not cfg.task.teacher_checkpoint:
            raise ValueError(f"{cfg.pipeline.name} experiments require task.teacher_checkpoint")
        if not cfg.task.prompt_bank_dir:
            raise ValueError(f"{cfg.pipeline.name} experiments require task.prompt_bank_dir")
        if cfg.task.subset_size <= 0:
            raise ValueError(f"{cfg.pipeline.name} experiments require task.subset_size > 0")
    if cfg.pipeline.name == "modadd_prompt_bank":
        if not cfg.task.prompt_bank_dir:
            raise ValueError("modadd_prompt_bank experiments require task.prompt_bank_dir")
        if cfg.task.n_train <= 0 or cfg.task.n_val <= 0:
            raise ValueError("modadd_prompt_bank experiments require task.n_train > 0 and task.n_val > 0")
    if cfg.pipeline.name == "modadd_render":
        if not cfg.task.teacher_checkpoint:
            raise ValueError("modadd_render experiments require task.teacher_checkpoint")
        if not cfg.task.prompt_bank_dir:
            raise ValueError("modadd_render experiments require task.prompt_bank_dir")
        if not cfg.task.dataset:
            raise ValueError("modadd_render experiments require task.dataset")
        if cfg.task.subset_size <= 0:
            raise ValueError("modadd_render experiments require task.subset_size > 0")
    return cfg
