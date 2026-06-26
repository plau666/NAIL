from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from nanogpt.config_schema import AppConfig, TorchrunConfig


@dataclass
class PretrainConfig:
    out_dir: str
    eval_interval: int
    log_interval: int
    eval_iters: int
    eval_only: bool
    always_save_checkpoint: bool
    init_from: str
    init_from_ckpt: str
    continue_from_subset_size: int
    wandb_log: bool
    wandb_project: str
    wandb_run_name: str | None
    wandb_run_id: str | None
    wandb_init_timeout: int
    dataset: str
    modadd_p: int
    modadd_m: int
    gradient_accumulation_steps: int
    batch_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool
    learning_rate: float
    max_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    decay_lr: bool
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float
    backend: str
    device: str
    dtype: str
    compile: bool
    modadd_eval_metrics: bool
    modadd_eval_clean_train_loss: bool
    eval_n: int
    eval_batch_size: int
    eval_seed: int
    save_every: int
    offline_single_epoch: bool
    offline_eval_full: bool
    offline_train_subset_size: int
    offline_train_shuffle: bool
    offline_target_type: str
    final_eval_on_exit: bool
    torchrun: TorchrunConfig
    python_bin: str

    def launcher_command(self) -> list[str]:
        return [self.python_bin, "-m", "nanogpt.run", *sys.argv[1:]]

    def worker_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("torchrun", None)
        payload.pop("python_bin", None)
        return payload


@dataclass
class StudentPrefixConfig:
    # Historical pipeline/default-rollout family (`nail` or `opd`). Paper method
    # names are resolved later from this plus loss and rollout temperature.
    method_family: str
    task: str
    teacher_checkpoint: str
    prompt_bank_dir: str
    subset_size: int
    eta: float
    teacher_law: str
    random_suffix_noise: dict[str, object]
    teacher_signal: str
    loss: str
    kl_beta: float | None
    out_dir: str
    init_from: str
    init_from_ckpt: str | None
    continue_from_subset_size: int
    batch_size: int
    max_iters: int
    learning_rate: float
    decay_lr: bool
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    rollout_temperature_override: float | None
    loss_temperature_override: float | None
    eval_interval: int
    eval_n: int
    eval_batch_size: int
    log_interval: int
    save_interval: int
    single_epoch: bool
    shuffle_prompts: bool
    seed: int
    device: str | None
    dtype: str | None
    compile: bool
    eps: float
    wandb_log: bool
    wandb_project: str
    wandb_run_name: str | None
    wandb_run_id: str | None
    wandb_init_timeout: int

    def config_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class OpdConfig(StudentPrefixConfig):
    pass


@dataclass
class NailConfig(StudentPrefixConfig):
    pass


def _require_model_field(value, field_name: str):
    if value is None:
        raise ValueError(f"pretrain experiments require model.{field_name}")
    return value


def project_pretrain_config(cfg: AppConfig) -> PretrainConfig:
    if cfg.pipeline.name != "pretrain":
        raise ValueError(f"expected pipeline='pretrain', got {cfg.pipeline.name!r}")
    if cfg.runtime.backend not in {"nccl", "gloo"}:
        raise ValueError(f"unsupported runtime.backend {cfg.runtime.backend!r}")
    if cfg.runtime.device == "cpu" and cfg.runtime.backend != "gloo":
        raise ValueError("runtime=cpu requires runtime.backend=gloo")
    if "cuda" in cfg.runtime.device and cfg.runtime.backend != "nccl":
        raise ValueError("CUDA runtimes require runtime.backend=nccl")
    if cfg.optim.continue_from_subset_size < 0:
        raise ValueError("continue_from_subset_size must be non-negative")
    if cfg.optim.init_from == "warm_start":
        if not cfg.optim.init_from_ckpt:
            raise ValueError("init_from='warm_start' requires init_from_ckpt")
        if cfg.optim.continue_from_subset_size > 0:
            if not cfg.task.dataset.startswith(("modadd_clean_offline", "modadd_noisy_offline")):
                raise ValueError(
                    "continue_from_subset_size is only supported for modular-addition offline datasets"
                )
            if not cfg.optim.offline_single_epoch:
                raise ValueError("continue_from_subset_size requires offline_single_epoch=True")
            if cfg.optim.offline_train_shuffle:
                raise ValueError("continue_from_subset_size requires offline_train_shuffle=False")
    elif cfg.optim.init_from_ckpt:
        raise ValueError("init_from_ckpt is only supported when init_from='warm_start'")

    python_bin = cfg.runtime.python_bin or sys.executable
    return PretrainConfig(
        out_dir=cfg.run.out_dir,
        eval_interval=cfg.optim.eval_interval,
        log_interval=cfg.optim.log_interval,
        eval_iters=cfg.optim.eval_iters,
        eval_only=cfg.optim.eval_only,
        always_save_checkpoint=cfg.optim.always_save_checkpoint,
        init_from=cfg.optim.init_from,
        init_from_ckpt=cfg.optim.init_from_ckpt,
        continue_from_subset_size=cfg.optim.continue_from_subset_size,
        wandb_log=cfg.logging.wandb_log,
        wandb_project=cfg.logging.wandb_project,
        wandb_run_name=cfg.logging.wandb_run_name,
        wandb_run_id=cfg.logging.wandb_run_id,
        wandb_init_timeout=cfg.logging.wandb_init_timeout,
        dataset=cfg.task.dataset,
        modadd_p=cfg.task.modadd_p,
        modadd_m=cfg.task.modadd_m,
        gradient_accumulation_steps=cfg.optim.gradient_accumulation_steps,
        batch_size=cfg.optim.batch_size,
        block_size=_require_model_field(cfg.model.block_size, "block_size"),
        n_layer=_require_model_field(cfg.model.n_layer, "n_layer"),
        n_head=_require_model_field(cfg.model.n_head, "n_head"),
        n_embd=_require_model_field(cfg.model.n_embd, "n_embd"),
        dropout=_require_model_field(cfg.model.dropout, "dropout"),
        bias=_require_model_field(cfg.model.bias, "bias"),
        learning_rate=cfg.optim.learning_rate,
        max_iters=cfg.optim.max_iters,
        weight_decay=cfg.optim.weight_decay,
        beta1=cfg.optim.beta1,
        beta2=cfg.optim.beta2,
        grad_clip=cfg.optim.grad_clip,
        decay_lr=cfg.optim.decay_lr,
        warmup_iters=cfg.optim.warmup_iters,
        lr_decay_iters=cfg.optim.lr_decay_iters,
        min_lr=cfg.optim.min_lr,
        backend=cfg.runtime.backend,
        device=cfg.runtime.device,
        dtype=cfg.runtime.dtype,
        compile=cfg.runtime.compile,
        modadd_eval_metrics=cfg.optim.modadd_eval_metrics,
        modadd_eval_clean_train_loss=cfg.optim.modadd_eval_clean_train_loss,
        eval_n=cfg.optim.eval_n,
        eval_batch_size=cfg.optim.eval_batch_size,
        eval_seed=cfg.optim.eval_seed,
        save_every=cfg.optim.save_every,
        offline_single_epoch=cfg.optim.offline_single_epoch,
        offline_eval_full=cfg.optim.offline_eval_full,
        offline_train_subset_size=cfg.optim.offline_train_subset_size,
        offline_train_shuffle=cfg.optim.offline_train_shuffle,
        offline_target_type=cfg.optim.offline_target_type,
        final_eval_on_exit=cfg.optim.final_eval_on_exit,
        torchrun=cfg.runtime.torchrun,
        python_bin=python_bin,
    )


def _project_student_prefix_config(
    cfg: AppConfig,
    *,
    expected_pipeline: str | Iterable[str],
    method_family: str,
    config_cls: type[StudentPrefixConfig],
) -> StudentPrefixConfig:
    expected_names = (
        {expected_pipeline}
        if isinstance(expected_pipeline, str)
        else set(expected_pipeline)
    )
    if cfg.pipeline.name not in expected_names:
        expected_text = "/".join(sorted(expected_names))
        raise ValueError(f"expected pipeline={expected_text!r}, got {cfg.pipeline.name!r}")
    if cfg.runtime.torchrun.nproc_per_node != 1 or cfg.runtime.torchrun.nnodes != 1:
        pipeline_label = cfg.pipeline.name.upper()
        raise ValueError(
            f"{pipeline_label} pipelines are single-process only; leave "
            "runtime.torchrun at 1"
        )
    if cfg.task.objective not in (None, ""):
        raise ValueError(
            "Student-prefix pipelines use task.loss and task.teacher_signal; "
            "task.objective is only supported for legacy metadata parsing."
        )
    return config_cls(
        method_family=method_family,
        task=cfg.task.task,
        teacher_checkpoint=cfg.task.teacher_checkpoint,
        prompt_bank_dir=cfg.task.prompt_bank_dir,
        subset_size=cfg.task.subset_size,
        eta=cfg.task.eta,
        teacher_law=cfg.task.teacher_law,
        random_suffix_noise=asdict(cfg.task.random_suffix_noise),
        teacher_signal=cfg.task.teacher_signal,
        loss=cfg.task.loss,
        kl_beta=cfg.task.kl_beta,
        out_dir=cfg.run.out_dir,
        init_from=cfg.optim.init_from,
        init_from_ckpt=cfg.optim.init_from_ckpt or None,
        continue_from_subset_size=cfg.optim.continue_from_subset_size,
        batch_size=cfg.optim.batch_size,
        max_iters=cfg.optim.max_iters,
        learning_rate=cfg.optim.learning_rate,
        decay_lr=cfg.optim.decay_lr,
        warmup_iters=cfg.optim.warmup_iters,
        lr_decay_iters=cfg.optim.lr_decay_iters,
        min_lr=cfg.optim.min_lr,
        weight_decay=cfg.optim.weight_decay,
        beta1=cfg.optim.beta1,
        beta2=cfg.optim.beta2,
        grad_clip=cfg.optim.grad_clip,
        rollout_temperature_override=cfg.task.rollout_temperature_override,
        loss_temperature_override=cfg.task.loss_temperature_override,
        eval_interval=cfg.optim.eval_interval,
        eval_n=cfg.optim.eval_n,
        eval_batch_size=cfg.optim.eval_batch_size,
        log_interval=cfg.optim.log_interval,
        save_interval=cfg.optim.save_interval,
        single_epoch=cfg.optim.single_epoch,
        shuffle_prompts=cfg.optim.shuffle_prompts,
        seed=cfg.optim.seed,
        device=cfg.runtime.device,
        dtype=cfg.runtime.dtype,
        compile=cfg.runtime.compile,
        eps=cfg.optim.eps,
        wandb_log=cfg.logging.wandb_log,
        wandb_project=cfg.logging.wandb_project,
        wandb_run_name=cfg.logging.wandb_run_name,
        wandb_run_id=cfg.logging.wandb_run_id,
        wandb_init_timeout=cfg.logging.wandb_init_timeout,
    )


def project_opd_config(cfg: AppConfig) -> OpdConfig:
    return _project_student_prefix_config(
        cfg,
        expected_pipeline="opd",
        method_family="opd",
        config_cls=OpdConfig,
    )


def project_nail_config(cfg: AppConfig) -> NailConfig:
    return _project_student_prefix_config(
        cfg,
        expected_pipeline={"nail", "student_prefix"},
        method_family="nail",
        config_cls=NailConfig,
    )
