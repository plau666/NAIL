from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic.prompt_bank import PromptBank, build_xy_from_prompt_and_target
from data.synthetic.target_spans import canonical_target_len
from nanogpt_checkpoint import load_nanogpt_model
from torch_dtypes import DTYPE_LOOKUP


def resolve_torch_dtype(dtype_name: str | None, device: str | torch.device) -> torch.dtype:
    if dtype_name is not None:
        return DTYPE_LOOKUP[dtype_name]
    if "cuda" in str(device):
        return torch.float16
    return torch.float32


def load_native_teacher(
    teacher_checkpoint: str | Path,
    *,
    device: str | torch.device,
    dtype_name: str | None,
):
    torch_dtype = resolve_torch_dtype(dtype_name, device)
    model = load_nanogpt_model(
        teacher_checkpoint,
        map_location="cpu",
        device=device,
        eval_mode=True,
    )
    model = model.to(device=device, dtype=torch_dtype)
    return model


ROLLOUT_MODE_CHOICES = ("greedy_then_corrupt", "sample_then_corrupt")
TARGET_MODE_CHOICES = ("tokens", "teacher_probs")


def resolve_offline_target_len(prompt_bank: PromptBank) -> int:
    return canonical_target_len(prompt_bank)


@torch.inference_mode()
def generate_teacher_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    target_mode: str = "tokens",
    device: str | torch.device,
    corrupt_ids_fn: Callable[[torch.Tensor, float], torch.Tensor],
    teacher_probs_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    teacher_probs_step_fn: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor] | None = None,
    rollout_probs_step_fn: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor] | None = None,
    saved_teacher_probs_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Roll out the frozen teacher to create fixed LogLossBC trajectories.

    This is the offline baseline in the paper: all prefixes and labels are
    generated before student training, unlike the student-prefix methods which
    query the teacher on current student prefixes. Because the realized noisy
    token is fed back into the next clean-teacher query, LogLossBC can train on
    downstream prefixes that are already corrupted or poisoned.
    """
    if rollout_mode not in ROLLOUT_MODE_CHOICES:
        raise ValueError(f"unknown rollout_mode={rollout_mode!r}")
    if target_mode not in TARGET_MODE_CHOICES:
        raise ValueError(f"unknown target_mode={target_mode!r}")
    if target_mode == "teacher_probs":
        if rollout_mode != "sample_then_corrupt" and rollout_probs_step_fn is None:
            raise ValueError(
                "target_mode='teacher_probs' currently only supports "
                "rollout_mode='sample_then_corrupt' unless a per-step rollout "
                "probability callback is provided"
            )
        if teacher_probs_fn is None and teacher_probs_step_fn is None:
            raise ValueError(
                "teacher_probs_fn or teacher_probs_step_fn is required when "
                "target_mode='teacher_probs'"
            )

    input_ids = prompt_ids.to(device=device, dtype=torch.long, non_blocking=True)
    generated = torch.empty(
        (prompt_ids.size(0), target_len),
        dtype=prompt_ids.dtype,
        device=device,
    )
    teacher_probs = None
    if target_mode == "teacher_probs":
        teacher_probs = torch.empty(
            (prompt_ids.size(0), target_len, model.config.vocab_size),
            dtype=saved_teacher_probs_dtype,
            device="cpu",
        )
    past_key_values = None

    for step in range(target_len):
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if isinstance(outputs, tuple):
            step_logits, _, past_key_values = outputs
        elif hasattr(outputs, "logits") and hasattr(outputs, "past_key_values"):
            step_logits = outputs.logits
            past_key_values = outputs.past_key_values
        else:
            raise TypeError(
                "teacher model outputs must be either a tuple "
                "(logits, loss, past_key_values) or an object with "
                ".logits and .past_key_values"
            )
        step_logits = step_logits[:, -1:, :]
        if teacher_probs is not None:
            if teacher_probs_step_fn is not None:
                step_teacher_probs = teacher_probs_step_fn(step_logits, step, prompt_ids)
            elif teacher_probs_fn is not None:
                step_teacher_probs = teacher_probs_fn(step_logits)
            else:
                raise ValueError("teacher probability callback unexpectedly missing")
            teacher_probs[:, step, :] = step_teacher_probs.squeeze(1).to(
                device="cpu",
                dtype=saved_teacher_probs_dtype,
            )
        if rollout_probs_step_fn is not None:
            step_rollout_probs = rollout_probs_step_fn(step_logits, step, prompt_ids)
            next_ids = torch.multinomial(
                step_rollout_probs.squeeze(1),
                num_samples=1,
            ).squeeze(1)
        elif rollout_mode == "greedy_then_corrupt":
            next_ids = torch.argmax(step_logits[:, -1, :], dim=-1)
        else:
            probs = torch.softmax(step_logits[:, -1, :].float(), dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        if rollout_probs_step_fn is None:
            next_ids = corrupt_ids_fn(next_ids, eta)
        next_ids = next_ids.to(dtype=prompt_ids.dtype)
        generated[:, step] = next_ids
        input_ids = next_ids.unsqueeze(1).to(dtype=torch.long)

    return generated.to(device="cpu", dtype=prompt_ids.dtype), teacher_probs


def render_train_split(
    model,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    *,
    eta: float,
    rollout_mode: str,
    target_mode: str = "tokens",
    gen_batch_size: int,
    device: str | torch.device,
    corrupt_ids_fn: Callable[[torch.Tensor, float], torch.Tensor],
    teacher_probs_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    teacher_probs_step_fn: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor] | None = None,
    rollout_probs_step_fn: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    subset_size = int(subset_idx.numel())
    target_len = resolve_offline_target_len(prompt_bank)
    train_x = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.token_dtype)
    train_y = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.label_dtype)
    train_teacher_probs = None
    if target_mode == "teacher_probs":
        if teacher_probs_fn is None and teacher_probs_step_fn is None:
            raise ValueError(
                "teacher_probs_fn or teacher_probs_step_fn is required when "
                "target_mode='teacher_probs'"
            )
        train_teacher_probs = torch.empty(
            (subset_size, target_len, model.config.vocab_size),
            dtype=torch.float16,
        )

    for start in range(0, subset_size, gen_batch_size):
        end = min(start + gen_batch_size, subset_size)
        batch_idx = subset_idx[start:end]
        batch_prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, batch_idx)
        batch_target_ids, batch_teacher_probs = generate_teacher_targets(
            model,
            batch_prompt_ids,
            target_len=target_len,
            eta=eta,
            rollout_mode=rollout_mode,
            target_mode=target_mode,
            device=device,
            corrupt_ids_fn=corrupt_ids_fn,
            teacher_probs_fn=teacher_probs_fn,
            teacher_probs_step_fn=teacher_probs_step_fn,
            rollout_probs_step_fn=rollout_probs_step_fn,
        )
        batch_x, batch_y = build_xy_from_prompt_and_target(batch_prompt_ids, batch_target_ids)
        train_x[start:end] = batch_x
        train_y[start:end] = batch_y
        if train_teacher_probs is not None:
            if batch_teacher_probs is None:
                raise ValueError("batch_teacher_probs unexpectedly missing")
            train_teacher_probs[start:end] = batch_teacher_probs
        print(f"train: rendered {end}/{subset_size}")

    return train_x, train_y, train_teacher_probs


def build_oracle_val_split(prompt_bank: PromptBank) -> tuple[torch.Tensor, torch.Tensor]:
    return build_xy_from_prompt_and_target(
        prompt_bank.clean_val_prompt_ids,
        prompt_bank.clean_val_cot_ids,
    )


def save_rendered_dataset(
    *,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_teacher_probs: torch.Tensor | None,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    save_dir: str | Path,
    meta: dict[str, Any],
) -> None:
    # Save both packed train tensors and the clean prompt/target provenance.
    # The provenance lets analysis scripts verify that online and offline runs
    # used the same prompt-bank subset described in the paper appendix.
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_x, save_dir / "train_x.pt")
    torch.save(train_y, save_dir / "train_y.pt")
    if train_teacher_probs is not None:
        torch.save(train_teacher_probs, save_dir / "train_teacher_probs.pt")
    torch.save(val_x, save_dir / "val_x.pt")
    torch.save(val_y, save_dir / "val_y.pt")
    del train_x, train_y, train_teacher_probs, val_x, val_y

    clean_train_prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, subset_idx)
    clean_train_cot_ids = prompt_bank.clean_train_cot_ids.index_select(0, subset_idx)

    torch.save(subset_idx.to(device="cpu", dtype=torch.long), save_dir / "subset_indices.pt")
    torch.save(clean_train_prompt_ids, save_dir / "clean_train_prompt_ids.pt")
    torch.save(clean_train_cot_ids, save_dir / "clean_train_cot_ids.pt")
    torch.save(prompt_bank.clean_val_prompt_ids, save_dir / "clean_val_prompt_ids.pt")
    torch.save(prompt_bank.clean_val_cot_ids, save_dir / "clean_val_cot_ids.pt")

    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_dataset_meta(
    *,
    prompt_bank: PromptBank,
    prompt_bank_dir: str | Path,
    teacher_checkpoint: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    target_mode: str = "tokens",
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "task": prompt_bank.task,
        "p": prompt_bank.p,
        "m": prompt_bank.m,
        "prompt_len": prompt_bank.prompt_len,
        "cot_len": prompt_bank.cot_len,
        "target_len": prompt_bank.target_len,
        "final_answer_len": prompt_bank.final_answer_len,
        "answer_len": prompt_bank.answer_len,
        "target_span": prompt_bank.meta.get("target_span", "cot_with_final_answer_suffix"),
        "subset_size": subset_size,
        "eta": eta,
        "gen_batch_size": gen_batch_size,
        "device": str(device),
        "dtype": dtype_name,
        "seed": seed,
        "prompt_bank_dir": str(prompt_bank_dir),
        "teacher_checkpoint": str(teacher_checkpoint),
        "train_targets_source": "teacher_rollout_with_optional_eta_corruption",
        "train_decode_mode": rollout_mode,
        "train_target_type": target_mode,
        "val_targets_source": "fixed_clean_oracle",
        "nested_subset_order_saved": True,
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta


def render_offline_dataset(
    *,
    teacher_checkpoint: str | Path,
    prompt_bank_dir: str | Path,
    save_dir: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    target_mode: str = "tokens",
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    corrupt_ids_fn: Callable[[torch.Tensor, float], torch.Tensor],
    teacher_probs_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    teacher_probs_step_fn: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor] | None = None,
    rollout_probs_step_fn: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    model = load_native_teacher(
        teacher_checkpoint,
        device=device,
        dtype_name=dtype_name,
    )

    train_x, train_y, train_teacher_probs = render_train_split(
        model,
        prompt_bank,
        subset_idx,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        gen_batch_size=gen_batch_size,
        device=device,
        corrupt_ids_fn=corrupt_ids_fn,
        teacher_probs_fn=teacher_probs_fn,
        teacher_probs_step_fn=teacher_probs_step_fn,
        rollout_probs_step_fn=rollout_probs_step_fn,
    )
    val_x, val_y = build_oracle_val_split(prompt_bank)

    meta = build_dataset_meta(
        prompt_bank=prompt_bank,
        prompt_bank_dir=prompt_bank_dir,
        teacher_checkpoint=teacher_checkpoint,
        subset_size=subset_size,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        gen_batch_size=gen_batch_size,
        device=device,
        dtype_name=dtype_name,
        seed=seed,
        extra_meta=extra_meta,
    )
    save_rendered_dataset(
        prompt_bank=prompt_bank,
        subset_idx=subset_idx,
        train_x=train_x,
        train_y=train_y,
        train_teacher_probs=train_teacher_probs,
        val_x=val_x,
        val_y=val_y,
        save_dir=save_dir,
        meta=meta,
    )
