from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.modular_addition.task import (
    corruptible_token_ids as modadd_corruptible_token_ids,
    evaluate_saved_clean_modadd_metrics,
)
from data.s5_cot.task import CORRUPTIBLE_IDS as S5_CORRUPTIBLE_IDS
from data.s5_cot.task import evaluate_saved_clean_s5_metrics
from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    semantic_key_noise_config_from_obj,
)
from data.synthetic.random_suffix_noise import (
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
    random_suffix_noise_config_from_obj,
    validate_random_suffix_applies_to_task,
)
from data.synthetic.prompt_bank import load_prompt_bank, select_train_subset
from data.synthetic.target_spans import (
    canonical_target_len,
    print_prompt_bank_target_span_diagnostic,
)
from nanogpt.methods.student_prefix import (
    FixedPromptCycle,
    cached_teacher_token_probs,
    default_rollout_temperature,
    evaluate_clean_ce_loss,
    extract_answer_logits,
    format_temperature_tag,
    forward_kl_full_loss,
    forward_kl_simple_loss,
    jsd_mc_loss,
    normalize_student_prefix_method,
    mixed_kl_loss_from_components,
    reverse_kl_full_loss,
    reverse_kl_tm_loss,
    rollout_student,
    sample_student_aux_actions,
    sample_teacher_actions,
)
from nanogpt.trainers.configs import StudentPrefixConfig
from nanogpt.trainers.runtime import (
    build_autocast_context,
    build_grad_scaler,
    capture_rng_state,
    get_nanogpt_lr,
    resolve_device,
    resolve_dtype,
    restore_rng_state,
)
from nanogpt.trainers.wandb import maybe_init_wandb
from nanogpt.utils.repo import write_launch_metadata
from nanogpt_checkpoint import (
    build_nanogpt_model,
    load_nanogpt_checkpoint,
    load_nanogpt_model,
)

# Native single-process implementation backend for student-prefix experiments.
# Paper methods are presets over this backend: NAIL-F/R use greedy prefixes,
# OPD-F/R use sampled prefixes, and LogLossBC stays in `workers/pretrain_body.py`.
# Historical Hydra pipeline names (`nail`, `opd`) remain for compatibility; use
# `resolved_method_name` in run metadata for the paper-facing label.

def validate_config(cfg: StudentPrefixConfig) -> None:
    # `method_family` is a historical pipeline/default-rollout selector, not a
    # paper method name. `resolved_method_name` in metadata is the reader-facing
    # NAIL-F/R or OPD-F/R label.
    method_family = getattr(cfg, "method_family", None)
    teacher_signal = getattr(cfg, "teacher_signal", None)
    loss = getattr(cfg, "loss", None)
    init_from = getattr(cfg, "init_from", "scratch")
    init_from_ckpt = getattr(cfg, "init_from_ckpt", None)
    continue_from_subset_size = getattr(cfg, "continue_from_subset_size", 0)
    single_epoch = getattr(cfg, "single_epoch", False)
    shuffle_prompts = getattr(cfg, "shuffle_prompts", False)
    subset_size = getattr(cfg, "subset_size", 0)
    rollout_temperature_override = getattr(cfg, "rollout_temperature_override", None)
    loss_temperature_override = getattr(cfg, "loss_temperature_override", None)
    kl_beta = getattr(cfg, "kl_beta", None)
    teacher_law = getattr(cfg, "teacher_law", "distributional_noise")
    task_name = getattr(cfg, "task", "s5")

    if method_family not in {"opd", "nail"}:
        raise ValueError(f"unknown method_family {method_family!r}")
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        if task_name != "s5":
            raise ValueError("semantic_key_noise teacher_law is only supported for S5")
        semantic_key_noise_config_from_obj(getattr(cfg, "semantic_key_noise", None))
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        random_suffix_config = random_suffix_noise_config_from_obj(
            getattr(cfg, "random_suffix_noise", None)
        )
        validate_random_suffix_applies_to_task(random_suffix_config, task_name=task_name)
    if teacher_signal not in {"mc", "full"}:
        raise ValueError("teacher_signal must be one of {'mc', 'full'}.")
    if loss not in {"forward", "reverse", "mixed", "jsd"}:
        raise ValueError("loss must be one of {'forward', 'reverse', 'mixed', 'jsd'}.")
    if loss in {"mixed", "jsd"}:
        if method_family != "nail":
            raise ValueError(f"{loss} loss is only supported for NAIL.")
        if teacher_signal != "mc":
            raise ValueError(f"{loss} loss requires teacher_signal='mc'.")
        if kl_beta is None:
            raise ValueError(f"{loss} loss requires task.kl_beta.")
        beta = float(kl_beta)
        if beta < 0.0 or beta > 1.0:
            raise ValueError("task.kl_beta must be in [0, 1].")
    elif kl_beta is not None:
        raise ValueError("task.kl_beta is only supported when task.loss is 'mixed' or 'jsd'.")
    if method_family == "opd" and loss != "reverse":
        raise ValueError("OPD only supports reverse loss.")
    if rollout_temperature_override is not None and float(rollout_temperature_override) < 0:
        raise ValueError("rollout_temperature_override must be non-negative.")
    if loss_temperature_override is not None and float(loss_temperature_override) <= 0:
        raise ValueError("loss_temperature_override must be > 0 when set.")
    if loss_temperature_override is not None and loss not in {"forward", "mixed", "jsd"}:
        raise ValueError("loss_temperature_override is only supported for forward, mixed, or jsd loss.")
    if getattr(cfg, "compile", False) and not hasattr(torch, "compile"):
        raise ValueError("--compile requires a PyTorch build with torch.compile support.")
    if init_from == "warm_start" and not init_from_ckpt:
        raise ValueError("--init_from=warm_start requires --init_from_ckpt.")
    if init_from != "warm_start" and init_from_ckpt is not None:
        raise ValueError("--init_from_ckpt is only supported with --init_from=warm_start.")
    if continue_from_subset_size < 0:
        raise ValueError("--continue_from_subset_size must be non-negative.")
    if continue_from_subset_size > 0:
        if init_from != "warm_start":
            raise ValueError(
                "--continue_from_subset_size is only supported with --init_from=warm_start."
            )
        if not single_epoch:
            raise ValueError("--continue_from_subset_size requires --single_epoch.")
        if shuffle_prompts:
            raise ValueError("--continue_from_subset_size requires prompts to remain unshuffled.")
        if continue_from_subset_size > subset_size:
            raise ValueError(
                "--continue_from_subset_size cannot exceed the current --subset_size."
            )


def normalize_state_dict_for_save(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


CLIPPING_FRACTION_EMA_DECAY = 0.95


def metric_scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().item())


def metric_mean(value: torch.Tensor) -> float:
    return metric_scalar(value.detach().float().mean())


def build_teacher_prob_kwargs(
    cfg: StudentPrefixConfig,
    *,
    clean_target_ids: torch.Tensor,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if cfg.teacher_law == SEMANTIC_KEY_NOISE_LAW:
        kwargs["semantic_key_noise_config"] = cfg.semantic_key_noise
    if cfg.teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        kwargs["clean_target_ids"] = clean_target_ids
        kwargs["random_suffix_noise_config"] = cfg.random_suffix_noise
        kwargs["task_name"] = cfg.task
    return kwargs


def build_reverse_mc_step_metrics(
    *,
    loss: torch.Tensor,
    objective_stats: dict[str, torch.Tensor],
    log_q: torch.Tensor,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    importance_weight = objective_stats["importance_weight"]
    metrics = {
        "train/loss": metric_scalar(loss),
        "train/advantage": metric_mean(objective_stats["advantage"]),
        "train/log_q": metric_mean(log_q),
        "train/log_teacher": metric_mean(objective_stats["log_teacher"]),
        "train/log_p": metric_mean(objective_stats["log_p"]),
        "train/importance_weight_mean": metric_mean(importance_weight),
        "train/importance_weight_std": metric_scalar(importance_weight.detach().float().std(unbiased=False)),
        "train/importance_weight_max": metric_scalar(importance_weight.detach().float().max()),
        "train/importance_weight_min": metric_scalar(importance_weight.detach().float().min()),
    }
    if extra_metrics is not None:
        metrics.update(extra_metrics)
    return metrics


def select_reverse_mc_actions(
    *,
    method_family: str,
    rollout_actions: torch.Tensor,
    rollout_log_q: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Choose which student action distribution drives the reverse-KL estimator.

    NAIL-R keeps the greedy rollout token only for prefix construction and draws
    an auxiliary token from the student distribution at that fixed prefix.
    OPD-R reuses the sampled rollout token as the reverse-KL sample; this is the
    literal MC estimator when rollout sampling matches the temperature-one
    student distribution, and a stopped-prefix surrogate otherwise.
    """
    if method_family == "opd":
        return rollout_actions, rollout_log_q, {}
    if method_family != "nail":
        raise ValueError(f"unknown method_family {method_family!r}")

    aux_actions, aux_log_q = sample_student_aux_actions(student_logits)
    return aux_actions, aux_log_q, {
        "train/rollout_log_q_mean": metric_mean(rollout_log_q),
        "train/aux_log_q_mean": metric_mean(aux_log_q),
        "train/aux_equals_rollout_rate": metric_mean(
            aux_actions.eq(rollout_actions).to(dtype=torch.float32)
        ),
    }


def _compute_total_grad_norm(model: torch.nn.Module) -> float:
    total_sq_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total_sq_norm += float(torch.sum(grad * grad).item())
    return total_sq_norm ** 0.5


def _compute_total_param_norm(model: torch.nn.Module) -> float:
    total_sq_norm = 0.0
    for param in model.parameters():
        value = param.detach().float()
        total_sq_norm += float(torch.sum(value * value).item())
    return total_sq_norm ** 0.5


def apply_grad_clip_with_diagnostics(
    model: torch.nn.Module,
    *,
    grad_clip: float,
) -> dict[str, float]:
    with torch.no_grad():
        pre_clip_grad_norm = _compute_total_grad_norm(model)
        param_norm = _compute_total_param_norm(model)
        grad_clipped = float(grad_clip > 0 and pre_clip_grad_norm > float(grad_clip))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            post_clip_grad_norm = _compute_total_grad_norm(model)
        else:
            post_clip_grad_norm = pre_clip_grad_norm
    return {
        "train/pre_clip_grad_norm": pre_clip_grad_norm,
        "train/post_clip_grad_norm": post_clip_grad_norm,
        "train/grad_clipped": grad_clipped,
        "train/param_norm": param_norm,
    }


def save_run_metadata(
    out_dir: Path,
    *,
    subset_indices: torch.Tensor,
    metadata: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(subset_indices.to(device="cpu", dtype=torch.long), out_dir / "subset_indices.pt")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def canonical_student_prefix_metadata(
    metadata: dict[str, object],
    *,
    default_method_family: str | None = None,
) -> dict[str, object]:
    task_name = metadata.get("task", "s5")
    method_state = normalize_student_prefix_method(
        metadata,
        default_method_family=default_method_family,
    )
    canonical = {
        "task": task_name,
        "p": metadata.get("p", 5 if task_name == "s5" else None),
        "m": metadata.get("m"),
        "teacher_checkpoint": metadata.get("teacher_checkpoint"),
        "prompt_bank_dir": metadata.get("prompt_bank_dir"),
        "subset_size": metadata.get("subset_size"),
        "eta": metadata.get("eta"),
        "teacher_law": metadata.get("teacher_law"),
        "semantic_key_noise": (
            metadata.get("semantic_key_noise")
            if metadata.get("teacher_law") == SEMANTIC_KEY_NOISE_LAW
            else None
        ),
        "random_suffix_noise": (
            metadata.get("random_suffix_noise")
            if metadata.get("teacher_law") == RANDOM_SUFFIX_AFTER_ERROR_LAW
            else None
        ),
        "method_family": method_state["method_family"],
        "implementation_backend": method_state["implementation_backend"],
        "resolved_method_name": method_state["resolved_method_name"],
        "resolved_rollout_policy": method_state["resolved_rollout_policy"],
        "teacher_signal": method_state["teacher_signal"],
        "loss": method_state["loss"],
        "kl_beta": method_state["kl_beta"],
        "resolved_rollout_temperature": method_state["resolved_rollout_temperature"],
        "resolved_loss_temperature": method_state["resolved_loss_temperature"],
        "shuffle_prompts": metadata.get("shuffle_prompts", False),
        "single_epoch": metadata.get("single_epoch", False),
        "init_from_ckpt": metadata.get("init_from_ckpt"),
        "continue_from_subset_size": metadata.get("continue_from_subset_size", 0),
        "seed": metadata.get("seed"),
    }
    if canonical["m"] is None and task_name == "s5":
        canonical["m"] = metadata.get("m")
    return canonical


def validate_resume_metadata(
    out_dir: Path,
    metadata: dict[str, object],
    *,
    default_method_family: str,
) -> None:
    meta_path = out_dir / "run_meta.json"
    if not meta_path.exists():
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    current_canonical = canonical_student_prefix_metadata(
        metadata,
        default_method_family=default_method_family,
    )
    saved_canonical = canonical_student_prefix_metadata(saved)
    for key, current_value in current_canonical.items():
        saved_value = saved_canonical.get(key)
        if saved_value != current_value:
            raise ValueError(
                f"Resume mismatch for {key}: saved={saved_value!r} current={current_value!r}"
            )


def mark_complete(out_dir: Path, iter_num: int) -> None:
    with open(out_dir / "completed.txt", "w", encoding="utf-8") as f:
        f.write(f"iter_num={iter_num}\n")


def load_student(
    args: StudentPrefixConfig,
    *,
    device: str,
) -> tuple[
    torch.nn.Module,
    dict[str, object],
    int,
    float,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    if args.init_from == "resume":
        model, checkpoint = load_nanogpt_model(
            args.out_dir,
            map_location="cpu",
            device=device,
            eval_mode=False,
            return_checkpoint=True,
        )
        iter_num = int(checkpoint["iter_num"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        return (
            model,
            dict(checkpoint["model_args"]),
            iter_num,
            best_val_loss,
            checkpoint.get("prompt_cycle_state"),
            checkpoint.get("rng_state"),
            checkpoint,
        )
    if args.init_from == "warm_start":
        model, checkpoint = load_nanogpt_model(
            args.init_from_ckpt,
            map_location="cpu",
            device=device,
            eval_mode=False,
            return_checkpoint=True,
        )
        iter_num = int(checkpoint["iter_num"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        return (
            model,
            dict(checkpoint["model_args"]),
            iter_num,
            best_val_loss,
            checkpoint.get("prompt_cycle_state"),
            checkpoint.get("rng_state"),
            checkpoint,
        )

    teacher_checkpoint = load_nanogpt_checkpoint(args.teacher_checkpoint, map_location="cpu")
    model_args = dict(teacher_checkpoint["model_args"])
    model = build_nanogpt_model(model_args)
    model.to(device)
    model.train()
    return model, model_args, 0, float("inf"), None, None, None


def maybe_hydrate_temperature_overrides(
    cfg: StudentPrefixConfig,
    *,
    source_config: dict[str, object] | None,
) -> None:
    if source_config is None:
        return
    source_state = normalize_student_prefix_method(source_config)
    if cfg.rollout_temperature_override is None:
        default_rollout = default_rollout_temperature(cfg.method_family)
        source_rollout = float(source_state["resolved_rollout_temperature"])
        if source_rollout != default_rollout:
            cfg.rollout_temperature_override = source_rollout
    if cfg.loss_temperature_override is None:
        source_loss_temp = source_state["resolved_loss_temperature"]
        if source_loss_temp is not None:
            cfg.loss_temperature_override = float(source_loss_temp)


def validate_warm_start_checkpoint(
    args: StudentPrefixConfig,
    *,
    subset_indices: torch.Tensor,
    checkpoint: dict[str, object],
) -> None:
    source_config = checkpoint.get("config", {})
    current_canonical = canonical_student_prefix_metadata(
        args.config_dict(),
        default_method_family=args.method_family,
    )
    source_canonical = canonical_student_prefix_metadata(source_config)
    for key in (
        "task",
        "eta",
        "teacher_law",
        "method_family",
        "teacher_signal",
        "loss",
        "resolved_rollout_temperature",
        "resolved_loss_temperature",
        "shuffle_prompts",
        "single_epoch",
    ):
        if source_canonical.get(key) != current_canonical.get(key):
            raise ValueError(
                f"Warm-start mismatch for {key}: checkpoint has {source_canonical.get(key)!r}, "
                f"current config requests {current_canonical.get(key)!r}"
            )

    if args.continue_from_subset_size <= 0:
        return

    prompt_cycle_state = checkpoint.get("prompt_cycle_state")
    if prompt_cycle_state is None:
        raise ValueError(
            "Warm-start continuation requires prompt_cycle_state in the source checkpoint."
        )

    expected_seen = int(args.continue_from_subset_size)
    source_n = int(prompt_cycle_state["n"])
    source_pos = int(prompt_cycle_state["pos"])
    source_epoch = int(prompt_cycle_state["epoch"])
    if source_n != expected_seen:
        raise ValueError(
            f"Warm-start source checkpoint covered n={source_n}, expected "
            f"continue_from_subset_size={expected_seen}"
        )
    if source_pos != expected_seen or source_epoch != 0:
        raise ValueError(
            "Warm-start source checkpoint must represent a completed single pass "
            f"through the first {expected_seen} prompts; got pos={source_pos}, epoch={source_epoch}"
        )

    source_subset_size = source_config.get("subset_size")
    if source_subset_size is not None and int(source_subset_size) != expected_seen:
        raise ValueError(
            f"Warm-start source checkpoint subset_size={source_subset_size} does not match "
            f"continue_from_subset_size={expected_seen}"
        )

    source_base_order = prompt_cycle_state["base_order"].to(device="cpu", dtype=torch.long)
    expected_base_order = subset_indices[:expected_seen].to(device="cpu", dtype=torch.long)
    if source_base_order.numel() != expected_seen or not torch.equal(source_base_order, expected_base_order):
        raise ValueError(
            "Warm-start source checkpoint does not match the prefix of the requested larger subset."
        )


def resolve_task_helpers(task_name: str, *, p: int):
    if task_name == "s5":
        return tuple(S5_CORRUPTIBLE_IDS), evaluate_saved_clean_s5_metrics
    if task_name == "modadd":
        return tuple(modadd_corruptible_token_ids(p)), evaluate_saved_clean_modadd_metrics
    raise ValueError(f"unknown task {task_name!r}")


def resolve_student_prefix_target_len(prompt_bank) -> int:
    return canonical_target_len(prompt_bank)


def validate_target_tensors(
    *,
    method_name: str,
    target_len: int,
    rollout_actions: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_target_logits: torch.Tensor,
) -> None:
    if int(rollout_actions.size(1)) != target_len:
        raise ValueError(
            f"{method_name} rollout action length {int(rollout_actions.size(1))} "
            f"does not match target_len {target_len}"
        )
    if int(teacher_probs.size(1)) != target_len:
        raise ValueError(
            f"{method_name} teacher_probs target length {int(teacher_probs.size(1))} "
            f"does not match target_len {target_len}"
        )
    if int(student_target_logits.size(1)) != target_len:
        raise ValueError(
            f"{method_name} student logits target length {int(student_target_logits.size(1))} "
            f"does not match target_len {target_len}"
        )


def save_eval_summary(out_dir: Path, *, iter_num: int, reason: str, metrics: dict[str, float]) -> None:
    payload = {
        "iter": int(iter_num),
        "reason": reason,
        **{key: float(value) for key, value in metrics.items()},
    }
    with open(out_dir / "last_eval.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "eval_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_eval(
    *,
    model,
    prompt_bank,
    prompt_bank_dir: str,
    task_name: str,
    device: str,
    autocast_context,
    eval_n: int,
    eval_batch_size: int,
) -> dict[str, float]:
    model.eval()
    val_loss = evaluate_clean_ce_loss(
        model,
        prompt_bank,
        batch_size=eval_batch_size,
        device=device,
        autocast_context=autocast_context,
    )
    _, evaluate_saved_metrics = resolve_task_helpers(task_name, p=prompt_bank.p)
    metrics = evaluate_saved_metrics(
        model,
        device=device,
        data_dir=prompt_bank_dir,
        n_eval=eval_n,
        batch_size=eval_batch_size,
    )
    model.train()
    eval_stats = {
        "val/loss": val_loss,
        "val/clean_full_exact": metrics["clean_full_exact"],
        "val/clean_final_exact": metrics["clean_final_exact"],
    }
    return eval_stats


def save_checkpoint(
    *,
    out_dir: Path,
    model,
    optimizer,
    model_args: dict[str, object],
    iter_num: int,
    best_val_loss: float,
    config: dict[str, object],
    prompt_cycle: FixedPromptCycle,
    device: str,
    clipping_fraction_ema: float,
) -> None:
    checkpoint = {
        "model": normalize_state_dict_for_save(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
        "prompt_cycle_state": prompt_cycle.state_dict(),
        "rng_state": capture_rng_state(device),
        "diagnostics_state": {
            "clipping_fraction_ema": float(clipping_fraction_ema),
        },
    }
    torch.save(checkpoint, out_dir / "ckpt.pt")


def build_wandb_name(
    cfg: StudentPrefixConfig,
    *,
    prompt_bank,
    method_state: dict[str, Any],
) -> str:
    eta_tag = str(cfg.eta).replace(".", "p")
    name = (
        f"{cfg.task}-{cfg.method_family}-{cfg.loss}-{cfg.teacher_signal}-"
        f"p{prompt_bank.p}-m{prompt_bank.m}-n{cfg.subset_size}-eta{eta_tag}-"
        f"{cfg.teacher_law}"
    )
    temp_tags: list[str] = []
    default_rollout = default_rollout_temperature(cfg.method_family)
    resolved_rollout = float(method_state["resolved_rollout_temperature"])
    resolved_loss = method_state["resolved_loss_temperature"]
    if resolved_rollout != default_rollout:
        temp_tags.append(f"roll{format_temperature_tag(resolved_rollout)}")
    if resolved_loss is not None:
        temp_tags.append(f"loss{format_temperature_tag(resolved_loss)}")
    if temp_tags:
        name += "-" + "-".join(temp_tags)
    if cfg.kl_beta is not None:
        beta_tag = str(float(cfg.kl_beta)).replace(".", "p").replace("-", "neg")
        beta_prefix = "jsd_beta" if cfg.loss == "jsd" else "beta"
        name += f"-{beta_prefix}{beta_tag}"
    return name


def build_run_metadata(
    cfg: StudentPrefixConfig,
    *,
    prompt_bank,
    device: str,
    torch_dtype: torch.dtype,
    method_state: dict[str, Any],
) -> dict[str, object]:
    rollout_temperature = float(method_state["resolved_rollout_temperature"])
    reverse_action_source = None
    if cfg.loss in {"mixed", "jsd"} and cfg.teacher_signal == "mc":
        beta = float(cfg.kl_beta)
        reverse_action_source = "student_aux_sample" if beta > 0.0 else None
    elif cfg.loss == "reverse" and cfg.teacher_signal == "mc":
        reverse_action_source = "student_aux_sample" if cfg.method_family == "nail" else "rollout_action"
    elif cfg.loss == "reverse" and cfg.teacher_signal == "full":
        reverse_action_source = "full_distribution"

    metadata = {
        "task": cfg.task,
        "p": prompt_bank.p,
        "m": prompt_bank.m,
        "prompt_len": prompt_bank.prompt_len,
        "cot_len": prompt_bank.cot_len,
        "target_len": prompt_bank.target_len,
        "final_answer_len": prompt_bank.final_answer_len,
        "answer_len": prompt_bank.answer_len,
        "target_span": prompt_bank.meta.get("target_span", "cot_with_final_answer_suffix"),
        "teacher_checkpoint": cfg.teacher_checkpoint,
        "prompt_bank_dir": cfg.prompt_bank_dir,
        "subset_size": cfg.subset_size,
        "eta": cfg.eta,
        "teacher_law": cfg.teacher_law,
        "implementation_backend": method_state["implementation_backend"],
        "resolved_method_name": method_state["resolved_method_name"],
        "method_family": cfg.method_family,
        "teacher_signal": cfg.teacher_signal,
        "loss": cfg.loss,
        "kl_beta": cfg.kl_beta,
        "rollout_temperature_override": cfg.rollout_temperature_override,
        "loss_temperature_override": cfg.loss_temperature_override,
        "resolved_rollout_temperature": rollout_temperature,
        "resolved_loss_temperature": method_state["resolved_loss_temperature"],
        "resolved_rollout_policy": method_state["resolved_rollout_policy"],
        "rollout_policy": method_state["resolved_rollout_policy"],
        "reverse_action_source": reverse_action_source,
        "shuffle_prompts": cfg.shuffle_prompts,
        "single_epoch": cfg.single_epoch,
        "init_from_ckpt": cfg.init_from_ckpt,
        "continue_from_subset_size": cfg.continue_from_subset_size,
        "seed": cfg.seed,
        "device": device,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "compile": bool(cfg.compile),
    }
    if cfg.teacher_law == SEMANTIC_KEY_NOISE_LAW:
        metadata["semantic_key_noise"] = semantic_key_noise_config_from_obj(
            cfg.semantic_key_noise
        ).to_dict()
    if cfg.teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        metadata["random_suffix_noise"] = random_suffix_noise_config_from_obj(
            cfg.random_suffix_noise
        ).to_dict()
    if "legacy_objective" in method_state:
        metadata["legacy_objective"] = method_state["legacy_objective"]
    return metadata


def run_student_prefix(cfg: StudentPrefixConfig, *, launcher_command: list[str]) -> None:
    """Train NAIL-F/R or OPD-F/R on learner-induced prefixes.

    This is the shared `student_prefix` backend. It samples or greedily collects
    prefixes under `torch.no_grad()`, queries the frozen noisy teacher on those
    fixed prefixes, and then computes the configured per-prefix loss. Gradients
    update only the current student logits used in the loss; they do not flow
    through rollout sampling, prompt selection, or teacher feedback.
    """
    validate_config(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            f"{cfg.method_family} native trainer is single-GPU only in v1."
        )

    device = resolve_device(cfg.device)
    torch_dtype = resolve_dtype(cfg.dtype, device)
    autocast_context = build_autocast_context(device, torch_dtype)
    scaler = build_grad_scaler(device=device, torch_dtype=torch_dtype)

    prompt_bank = load_prompt_bank(cfg.prompt_bank_dir)
    if prompt_bank.task != cfg.task:
        raise ValueError(
            f"Prompt bank task mismatch: prompt bank has task={prompt_bank.task!r} "
            f"but task={cfg.task!r}"
        )
    target_len = resolve_student_prefix_target_len(prompt_bank)
    training_seq_len = prompt_bank.prompt_len + target_len - 1
    print_prompt_bank_target_span_diagnostic(
        method_name=f"{cfg.method_family}/{cfg.loss}_{cfg.teacher_signal}",
        prompt_bank=prompt_bank,
        actual_target_len=target_len,
        total_sequence_len=training_seq_len,
        target_description=(
            "clean reference continuation; online teacher supervision is applied "
            "over the same target positions"
        ),
    )
    corruptible_ids, _ = resolve_task_helpers(cfg.task, p=prompt_bank.p)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    student, model_args, iter_num, best_val_loss, prompt_cycle_state, rng_state, source_checkpoint = load_student(
        cfg,
        device=device,
    )
    if source_checkpoint is not None:
        maybe_hydrate_temperature_overrides(
            cfg,
            source_config=source_checkpoint.get("config"),
        )
    method_state = normalize_student_prefix_method(
        cfg,
        default_method_family=cfg.method_family,
    )

    if cfg.wandb_run_name is None:
        cfg.wandb_run_name = build_wandb_name(
            cfg,
            prompt_bank=prompt_bank,
            method_state=method_state,
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_launch_metadata(out_dir, cfg=cfg, command=launcher_command)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    subset_indices = select_train_subset(prompt_bank, cfg.subset_size)
    run_metadata = build_run_metadata(
        cfg,
        prompt_bank=prompt_bank,
        device=device,
        torch_dtype=torch_dtype,
        method_state=method_state,
    )
    if cfg.init_from == "resume":
        validate_resume_metadata(
            out_dir,
            run_metadata,
            default_method_family=cfg.method_family,
        )
    save_run_metadata(out_dir, subset_indices=subset_indices, metadata=run_metadata)

    if cfg.init_from == "warm_start":
        assert source_checkpoint is not None
        validate_warm_start_checkpoint(
            cfg,
            subset_indices=subset_indices,
            checkpoint=source_checkpoint,
        )

    teacher = load_nanogpt_model(
        cfg.teacher_checkpoint,
        map_location="cpu",
        device=device,
        eval_mode=True,
    )
    for param in teacher.parameters():
        param.requires_grad = False

    prompt_cycle = FixedPromptCycle(
        prompt_bank.clean_train_prompt_ids,
        order=subset_indices,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_prompts,
        seed=cfg.seed,
    )
    if cfg.init_from == "resume" and prompt_cycle_state is not None:
        prompt_cycle.load_state_dict(prompt_cycle_state)
    elif cfg.init_from == "warm_start" and cfg.continue_from_subset_size > 0:
        prompt_cycle.pos = int(cfg.continue_from_subset_size)
    if rng_state is not None:
        restore_rng_state(rng_state, device)

    optimizer = student.configure_optimizers(
        cfg.weight_decay,
        cfg.learning_rate,
        (cfg.beta1, cfg.beta2),
        "cuda" if "cuda" in device else "cpu",
    )
    if cfg.init_from in {"resume", "warm_start"}:
        assert source_checkpoint is not None
        optimizer.load_state_dict(source_checkpoint["optimizer"])

    if cfg.compile:
        print("compiling the student model... (takes a ~minute)")
        student = torch.compile(student)

    config = cfg.config_dict()
    config.update(
        {
            "task": cfg.task,
            "p": prompt_bank.p,
            "m": prompt_bank.m,
            "resolved_device": device,
            "resolved_dtype": str(torch_dtype).replace("torch.", ""),
            "prompt_len": prompt_bank.prompt_len,
            "cot_len": prompt_bank.cot_len,
            "target_len": prompt_bank.target_len,
            "final_answer_len": prompt_bank.final_answer_len,
            "answer_len": prompt_bank.answer_len,
            "target_span": prompt_bank.meta.get("target_span", "cot_with_final_answer_suffix"),
            "implementation_backend": method_state["implementation_backend"],
            "resolved_method_name": method_state["resolved_method_name"],
            "resolved_rollout_policy": method_state["resolved_rollout_policy"],
            "teacher_signal": method_state["teacher_signal"],
            "loss": method_state["loss"],
            "kl_beta": method_state["kl_beta"],
            "resolved_rollout_temperature": method_state["resolved_rollout_temperature"],
            "resolved_loss_temperature": method_state["resolved_loss_temperature"],
        }
    )
    if "legacy_objective" in method_state:
        config["legacy_objective"] = method_state["legacy_objective"]
    wandb = maybe_init_wandb(
        enabled=cfg.wandb_log,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
        run_id=cfg.wandb_run_id,
        out_dir=cfg.out_dir,
        init_from=cfg.init_from,
        init_timeout=cfg.wandb_init_timeout,
        config=config,
    )

    rollout_temperature = float(method_state["resolved_rollout_temperature"])
    loss_temperature = method_state["resolved_loss_temperature"]
    # Rollout temperature controls prefix collection; loss temperature controls
    # the trained student distribution for forward/mixed/JSD losses.
    kl_beta = None if cfg.kl_beta is None else float(cfg.kl_beta)
    running_metrics: dict[str, float] = {}
    running_steps = 0
    clipping_fraction_ema = 0.0
    if source_checkpoint is not None:
        diagnostics_state = source_checkpoint.get("diagnostics_state") or {}
        clipping_fraction_ema = float(diagnostics_state.get("clipping_fraction_ema", 0.0))
    latest_step_metrics: dict[str, float] = {}
    t0 = time.time()

    while iter_num < cfg.max_iters:
        if cfg.single_epoch and not prompt_cycle.has_remaining_in_epoch():
            break

        lr = (
            get_nanogpt_lr(
                iter_num,
                learning_rate=cfg.learning_rate,
                warmup_iters=cfg.warmup_iters,
                lr_decay_iters=cfg.lr_decay_iters,
                min_lr=cfg.min_lr,
            )
            if cfg.decay_lr
            else cfg.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg.eval_interval == 0:
            eval_stats = run_eval(
                model=student,
                prompt_bank=prompt_bank,
                prompt_bank_dir=cfg.prompt_bank_dir,
                task_name=cfg.task,
                device=device,
                autocast_context=autocast_context,
                eval_n=cfg.eval_n,
                eval_batch_size=cfg.eval_batch_size,
            )
            best_val_loss = min(best_val_loss, eval_stats["val/loss"])
            save_eval_summary(out_dir, iter_num=iter_num, reason="periodic", metrics=eval_stats)
            msg_parts = [f"eval step {iter_num}: val loss {eval_stats['val/loss']:.4f}"]
            if "val/cot_exact" in eval_stats:
                msg_parts.append(f"val cot_exact {eval_stats['val/cot_exact']:.4f}")
            msg_parts.append(f"val clean_full_exact {eval_stats['val/clean_full_exact']:.4f}")
            msg_parts.append(f"val clean_final_exact {eval_stats['val/clean_final_exact']:.4f}")
            print(", ".join(msg_parts))
            save_checkpoint(
                out_dir=out_dir,
                model=student,
                optimizer=optimizer,
                model_args=model_args,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                config=config,
                prompt_cycle=prompt_cycle,
                device=device,
                clipping_fraction_ema=clipping_fraction_ema,
            )
            if cfg.save_interval > 0 and iter_num > 0 and iter_num % cfg.save_interval == 0:
                torch.save(
                    torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False),
                    out_dir / f"ckpt_{iter_num:07d}.pt",
                )
            if wandb is not None:
                wandb.log({"iter": iter_num, **eval_stats})

        prompt_indices = (
            prompt_cycle.next_batch_indices_no_wrap()
            if cfg.single_epoch
            else prompt_cycle.next_batch_indices()
        )
        prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, prompt_indices)
        clean_target_ids = prompt_bank.clean_train_cot_ids.index_select(
            0,
            prompt_indices,
        )[:, :target_len]

        teacher_targets = None
        log_teacher_target = None

        student.eval()
        with torch.no_grad():
            # Rollout may be greedy (NAIL-F/R) or sampled (OPD-F/R).
            # This stopped rollout defines the prefix distribution in the
            # paper's augmented-trajectory objectives. The no-grad boundary is
            # intentional: gradients below update the next-token distribution at
            # the visited prefixes, not the sampling process that produced them.
            full_seq, rollout_actions, rollout_log_q = rollout_student(
                student,
                prompt_ids,
                target_len=target_len,
                temperature=rollout_temperature,
                device=device,
                autocast_context=autocast_context,
            )
            rollout_inputs = full_seq[:, :-1]
            teacher_prob_kwargs = build_teacher_prob_kwargs(
                cfg,
                clean_target_ids=clean_target_ids,
            )
            teacher_probs = cached_teacher_token_probs(
                teacher,
                prompt_ids,
                rollout_actions,
                eta=cfg.eta,
                teacher_law=cfg.teacher_law,
                corruptible_token_ids=corruptible_ids,
                device=device,
                autocast_context=autocast_context,
                **teacher_prob_kwargs,
            )
            needs_forward_teacher_targets = (
                cfg.teacher_signal == "mc"
                and (
                    cfg.loss == "forward"
                    or (
                        cfg.loss in {"mixed", "jsd"}
                        and kl_beta is not None
                        and kl_beta < 1.0
                    )
                )
            )
            if needs_forward_teacher_targets:
                # NAIL-F / OPD-F MC targets come from the noisy expert law, not
                # from the student rollout token.
                teacher_targets = sample_teacher_actions(teacher_probs)
                teacher_target_probs = teacher_probs.gather(
                    2,
                    teacher_targets.unsqueeze(-1),
                ).squeeze(-1)
                log_teacher_target = torch.log(teacher_target_probs.clamp_min(cfg.eps))
        student.train()

        with autocast_context:
            p_logits, _ = student(rollout_inputs, return_full_logits=True)
            p_answer_logits = extract_answer_logits(
                p_logits,
                prompt_len=prompt_bank.prompt_len,
                target_len=target_len,
            )
            validate_target_tensors(
                method_name=f"{cfg.method_family}/{cfg.loss}_{cfg.teacher_signal}",
                target_len=target_len,
                rollout_actions=rollout_actions,
                teacher_probs=teacher_probs,
                student_target_logits=p_answer_logits,
            )
            if cfg.teacher_signal == "mc" and cfg.loss == "reverse":
                # OPD-R scores sampled rollout actions directly. NAIL-R keeps
                # those rollout actions only for prefix construction and samples
                # separate auxiliary student actions on the resulting fixed
                # prefixes. If OPD-R uses a non-temperature-one rollout, this
                # branch is a surrogate rather than the literal reverse-KL MC
                # estimator for the temperature-one student.
                reverse_actions, reverse_log_q, reverse_metrics = select_reverse_mc_actions(
                    method_family=cfg.method_family,
                    rollout_actions=rollout_actions,
                    rollout_log_q=rollout_log_q,
                    student_logits=p_answer_logits,
                )
                loss, objective_stats = reverse_kl_tm_loss(
                    p_answer_logits,
                    reverse_actions,
                    log_q=reverse_log_q,
                    teacher_probs=teacher_probs,
                    eps=cfg.eps,
                )
                step_metrics = build_reverse_mc_step_metrics(
                    loss=loss,
                    objective_stats=objective_stats,
                    log_q=reverse_log_q,
                    extra_metrics=reverse_metrics,
                )
            elif cfg.teacher_signal == "full" and cfg.loss == "reverse":
                loss, objective_stats = reverse_kl_full_loss(
                    p_answer_logits,
                    teacher_probs=teacher_probs,
                    eps=cfg.eps,
                )
                step_metrics = {
                    "train/loss": metric_scalar(loss),
                    "train/reverse_kl": metric_mean(objective_stats["reverse_kl"]),
                    "train/reverse_kl_full": metric_mean(objective_stats["reverse_kl"]),
                    "train/student_teacher_ce": metric_mean(objective_stats["student_teacher_ce"]),
                    "train/student_entropy": metric_mean(objective_stats["student_entropy"]),
                }
            elif cfg.teacher_signal == "mc" and cfg.loss == "mixed":
                assert kl_beta is not None
                # Paper Appendix B.3.2: beta interpolates between the forward
                # and reverse stopped-prefix estimators on greedy NAIL-F/R prefixes.
                forward_weight = 1.0 - kl_beta
                reverse_weight = kl_beta
                zero_loss = p_answer_logits.sum() * 0.0
                forward_loss = zero_loss
                reverse_loss = zero_loss
                step_metrics = {"train/mixed_beta": kl_beta}

                if forward_weight > 0.0:
                    assert teacher_targets is not None
                    assert log_teacher_target is not None
                    forward_loss, forward_stats = forward_kl_simple_loss(
                        p_answer_logits,
                        teacher_targets,
                        teacher_probs=teacher_probs,
                        temperature=loss_temperature,
                        eps=cfg.eps,
                    )
                    step_metrics["train/log_student_target"] = metric_mean(
                        forward_stats["log_student_target"]
                    )
                    step_metrics["train/log_teacher_target"] = metric_mean(log_teacher_target)

                if reverse_weight > 0.0:
                    reverse_actions, reverse_log_q, reverse_metrics = select_reverse_mc_actions(
                        method_family=cfg.method_family,
                        rollout_actions=rollout_actions,
                        rollout_log_q=rollout_log_q,
                        student_logits=p_answer_logits,
                    )
                    reverse_loss, reverse_stats = reverse_kl_tm_loss(
                        p_answer_logits,
                        reverse_actions,
                        log_q=reverse_log_q,
                        teacher_probs=teacher_probs,
                        eps=cfg.eps,
                    )
                    reverse_step_metrics = build_reverse_mc_step_metrics(
                        loss=reverse_loss,
                        objective_stats=reverse_stats,
                        log_q=reverse_log_q,
                        extra_metrics=reverse_metrics,
                    )
                    for key, value in reverse_step_metrics.items():
                        if key != "train/loss":
                            step_metrics[key] = value

                loss = mixed_kl_loss_from_components(
                    forward_loss,
                    reverse_loss,
                    beta=kl_beta,
                )
                step_metrics["train/loss"] = metric_scalar(loss)
                step_metrics["train/mixed_forward_loss"] = metric_scalar(forward_loss)
                step_metrics["train/mixed_reverse_loss"] = metric_scalar(reverse_loss)
            elif cfg.teacher_signal == "mc" and cfg.loss == "jsd":
                assert kl_beta is not None
                if kl_beta == 0.0:
                    assert teacher_targets is not None
                    assert log_teacher_target is not None
                    loss, objective_stats = forward_kl_simple_loss(
                        p_answer_logits,
                        teacher_targets,
                        teacher_probs=teacher_probs,
                        temperature=loss_temperature,
                        eps=cfg.eps,
                    )
                    step_metrics = {
                        "train/loss": metric_scalar(loss),
                        "train/jsd_beta": kl_beta,
                        "train/jsd_loss": metric_scalar(loss),
                        "train/jsd_teacher_to_mix_loss": metric_scalar(loss),
                        "train/jsd_student_to_mix_loss": 0.0,
                        "train/log_student_target": metric_mean(
                            objective_stats["log_student_target"]
                        ),
                        "train/log_teacher_target": metric_mean(log_teacher_target),
                    }
                elif kl_beta == 1.0:
                    reverse_actions, reverse_log_q, reverse_metrics = select_reverse_mc_actions(
                        method_family=cfg.method_family,
                        rollout_actions=rollout_actions,
                        rollout_log_q=rollout_log_q,
                        student_logits=p_answer_logits,
                    )
                    loss, objective_stats = reverse_kl_tm_loss(
                        p_answer_logits,
                        reverse_actions,
                        log_q=reverse_log_q,
                        teacher_probs=teacher_probs,
                        eps=cfg.eps,
                    )
                    step_metrics = build_reverse_mc_step_metrics(
                        loss=loss,
                        objective_stats=objective_stats,
                        log_q=reverse_log_q,
                        extra_metrics=reverse_metrics,
                    )
                    step_metrics["train/jsd_beta"] = kl_beta
                    step_metrics["train/jsd_loss"] = metric_scalar(loss)
                    step_metrics["train/jsd_teacher_to_mix_loss"] = 0.0
                    step_metrics["train/jsd_student_to_mix_loss"] = metric_scalar(loss)
                else:
                    assert teacher_targets is not None
                    reverse_actions, reverse_log_q, reverse_metrics = select_reverse_mc_actions(
                        method_family=cfg.method_family,
                        rollout_actions=rollout_actions,
                        rollout_log_q=rollout_log_q,
                        student_logits=p_answer_logits,
                    )
                    loss, objective_stats = jsd_mc_loss(
                        p_answer_logits,
                        teacher_targets,
                        reverse_actions,
                        teacher_probs=teacher_probs,
                        beta=kl_beta,
                        temperature=loss_temperature,
                        eps=cfg.eps,
                    )
                    step_metrics = {
                        "train/loss": metric_scalar(loss),
                        "train/jsd_beta": kl_beta,
                        "train/jsd_loss": metric_scalar(loss),
                        "train/jsd_teacher_to_mix_loss": metric_scalar(
                            objective_stats["teacher_to_mix_loss"]
                        ),
                        "train/jsd_student_to_mix_loss": metric_scalar(
                            objective_stats["student_to_mix_loss"]
                        ),
                        "train/log_teacher_target": metric_mean(
                            objective_stats["log_teacher_target"]
                        ),
                        "train/jsd_log_mixture_teacher_target": metric_mean(
                            objective_stats["log_mixture_teacher_target"]
                        ),
                        "train/jsd_log_student_action": metric_mean(
                            objective_stats["log_student_action"]
                        ),
                        "train/jsd_log_mixture_student_action": metric_mean(
                            objective_stats["log_mixture_student_action"]
                        ),
                        "train/jsd_teacher_to_mix_token": metric_mean(
                            objective_stats["teacher_to_mix"]
                        ),
                        "train/jsd_student_to_mix_token": metric_mean(
                            objective_stats["student_to_mix"]
                        ),
                        "train/log_q": metric_mean(reverse_log_q),
                    }
                    step_metrics.update(reverse_metrics)
            elif cfg.teacher_signal == "mc" and cfg.loss == "forward":
                # Forward loss on greedy prefixes is NAIL-F; the same loss on
                # sampled prefixes is OPD-F; legacy override-based launches
                # may still arrive through `pipeline=nail`.
                assert teacher_targets is not None
                assert log_teacher_target is not None
                loss, objective_stats = forward_kl_simple_loss(
                    p_answer_logits,
                    teacher_targets,
                    teacher_probs=teacher_probs,
                    temperature=loss_temperature,
                    eps=cfg.eps,
                )
                step_metrics = {
                    "train/loss": metric_scalar(loss),
                    "train/log_student_target": metric_mean(objective_stats["log_student_target"]),
                    "train/log_teacher_target": metric_mean(log_teacher_target),
                }
            else:
                # Full-distribution losses train all logits under the loss temperature.
                loss, objective_stats = forward_kl_full_loss(
                    p_answer_logits,
                    teacher_probs=teacher_probs,
                    temperature=loss_temperature,
                    eps=cfg.eps,
                )
                step_metrics = {
                    "train/loss": metric_scalar(loss),
                    "train/forward_kl": metric_mean(objective_stats["forward_kl"]),
                    "train/teacher_ce": metric_mean(objective_stats["teacher_ce"]),
                    "train/teacher_entropy": metric_mean(objective_stats["teacher_entropy"]),
                }

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        optimizer_step_metrics = apply_grad_clip_with_diagnostics(
            student,
            grad_clip=cfg.grad_clip,
        )
        clipping_fraction_ema = (
            CLIPPING_FRACTION_EMA_DECAY * clipping_fraction_ema
            + (1.0 - CLIPPING_FRACTION_EMA_DECAY) * optimizer_step_metrics["train/grad_clipped"]
        )
        latest_step_metrics = {
            **optimizer_step_metrics,
            "train/clipping_fraction_ema": float(clipping_fraction_ema),
            "train/lr": float(lr),
        }
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        for key, value in step_metrics.items():
            running_metrics[key] = running_metrics.get(key, 0.0) + value
        running_steps += 1

        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            denom = max(1, running_steps)
            train_stats = {key: value / denom for key, value in running_metrics.items()}
            train_stats.update(latest_step_metrics)
            train_stats["iter"] = iter_num
            metric_str = ", ".join(
                f"{key.split('/')[-1]} {train_stats[key]:.4f}"
                for key in train_stats
                if key.startswith("train/")
            )
            print(f"iter {iter_num}: {metric_str}, time {dt*1000:.2f}ms")
            if wandb is not None:
                wandb.log(train_stats)
            running_metrics = {}
            running_steps = 0

        iter_num += 1

    final_stats = run_eval(
        model=student,
        prompt_bank=prompt_bank,
        prompt_bank_dir=cfg.prompt_bank_dir,
        task_name=cfg.task,
        device=device,
        autocast_context=autocast_context,
        eval_n=cfg.eval_n,
        eval_batch_size=cfg.eval_batch_size,
    )
    best_val_loss = min(best_val_loss, final_stats["val/loss"])
    save_eval_summary(out_dir, iter_num=iter_num, reason="final", metrics=final_stats)
    final_msg_parts = [f"final step {iter_num}: val loss {final_stats['val/loss']:.4f}"]
    if "val/cot_exact" in final_stats:
        final_msg_parts.append(f"val cot_exact {final_stats['val/cot_exact']:.4f}")
    final_msg_parts.append(f"val clean_full_exact {final_stats['val/clean_full_exact']:.4f}")
    final_msg_parts.append(f"val clean_final_exact {final_stats['val/clean_final_exact']:.4f}")
    print(", ".join(final_msg_parts))
    save_checkpoint(
        out_dir=out_dir,
        model=student,
        optimizer=optimizer,
        model_args=model_args,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        config=config,
        prompt_cycle=prompt_cycle,
        device=device,
        clipping_fraction_ema=clipping_fraction_ema,
    )
    mark_complete(out_dir, iter_num)
    if wandb is not None:
        wandb.log({"iter": iter_num, **final_stats})
        wandb.finish()
