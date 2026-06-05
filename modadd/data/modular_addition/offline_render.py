from __future__ import annotations

from pathlib import Path

import torch

from data.modular_addition.prompt_bank import (
    PromptBank,
    build_xy_from_prompt_and_target,
    load_prompt_bank,
    select_train_subset,
)
from data.modular_addition.task import corrupt_ids, corruptible_token_ids
from data.synthetic.random_suffix_noise import (
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
    RandomSuffixStepSpec,
    generate_random_suffix_after_error_targets,
    make_random_suffix_generator,
    random_suffix_noise_config_from_obj,
    random_suffix_noise_meta,
    validate_random_suffix_applies_to_task,
)
from data.synthetic.offline_render import (
    ROLLOUT_MODE_CHOICES,
    build_dataset_meta as shared_build_dataset_meta,
    build_oracle_val_split,
    generate_teacher_targets as shared_generate_teacher_targets,
    load_native_teacher,
    render_offline_dataset as shared_render_offline_dataset,
    render_train_split as shared_render_train_split,
    save_rendered_dataset,
)


TEACHER_LAW_CHOICES = ("distributional_noise", RANDOM_SUFFIX_AFTER_ERROR_LAW)


def _build_random_suffix_step_spec_fn():
    # In modular addition every target position is a semantic running-sum
    # token, so the paper's absorbing law can treat every target token as a
    # possible poison trigger.
    def step_spec(step: int, prompt_ids: torch.Tensor, device: torch.device) -> RandomSuffixStepSpec:
        del step
        return RandomSuffixStepSpec(
            key_mask=torch.ones(prompt_ids.size(0), dtype=torch.bool, device=device),
            semantic_mask=torch.ones(prompt_ids.size(0), dtype=torch.bool, device=device),
            scaffold_token_ids=None,
        )

    return step_spec


def _generate_random_suffix_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    device: str | torch.device,
    p: int,
    random_suffix_noise_config=None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    # Offline rendering samples a full noisy expert trajectory for LogLossBC.
    # Student-prefix methods use the same law through `cached_teacher_token_probs`,
    # but infer poisoning from the student prefix instead of this render state.
    config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
    validate_random_suffix_applies_to_task(config, task_name="modadd")
    return generate_random_suffix_after_error_targets(
        model,
        prompt_ids,
        target_len=target_len,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode="tokens",
        device=device,
        config=config,
        eligible_token_ids=corruptible_token_ids(p),
        step_spec_fn=_build_random_suffix_step_spec_fn(),
        generator=generator,
    )


def _random_suffix_extra_meta(
    *,
    eta: float,
    rollout_mode: str,
    p: int,
    random_suffix_noise_config=None,
) -> dict[str, object]:
    config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
    validate_random_suffix_applies_to_task(config, task_name="modadd")
    return {
        "teacher_law": RANDOM_SUFFIX_AFTER_ERROR_LAW,
        "random_suffix_noise": random_suffix_noise_meta(
            config,
            eta=eta,
            task_name="modadd",
            eligible_token_ids=corruptible_token_ids(p),
        ),
        "requested_rollout_mode": rollout_mode,
        "train_decode_mode": f"{RANDOM_SUFFIX_AFTER_ERROR_LAW}_sample",
    }


def generate_teacher_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    device: str | torch.device,
    p: int,
    teacher_law: str = "distributional_noise",
    random_suffix_noise_config=None,
) -> torch.Tensor:
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        targets, _, _ = _generate_random_suffix_targets(
            model,
            prompt_ids,
            target_len=target_len,
            eta=eta,
            rollout_mode=rollout_mode,
            device=device,
            p=p,
            random_suffix_noise_config=random_suffix_noise_config,
        )
        return targets
    targets, _ = shared_generate_teacher_targets(
        model,
        prompt_ids,
        target_len=target_len,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode="tokens",
        device=device,
        corrupt_ids_fn=lambda ids, noise: corrupt_ids(ids, noise, p=p),
    )
    return targets


def render_train_split(
    model,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    *,
    eta: float,
    rollout_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    teacher_law: str = "distributional_noise",
    random_suffix_noise_config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        train_x, train_y, _ = _render_random_suffix_train_split(
            model,
            prompt_bank,
            subset_idx,
            eta=eta,
            rollout_mode=rollout_mode,
            gen_batch_size=gen_batch_size,
            device=device,
            random_suffix_noise_config=random_suffix_noise_config,
        )
        return train_x, train_y
    train_x, train_y, _ = shared_render_train_split(
        model,
        prompt_bank,
        subset_idx,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode="tokens",
        gen_batch_size=gen_batch_size,
        device=device,
        corrupt_ids_fn=lambda ids, noise: corrupt_ids(ids, noise, p=prompt_bank.p),
    )
    return train_x, train_y


def build_dataset_meta(
    *,
    prompt_bank: PromptBank,
    prompt_bank_dir: str | Path,
    teacher_checkpoint: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
    teacher_law: str = "distributional_noise",
    random_suffix_noise_config=None,
) -> dict[str, object]:
    extra_meta: dict[str, object] = {"teacher_law": teacher_law}
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        extra_meta.update(
            _random_suffix_extra_meta(
                eta=eta,
                rollout_mode=rollout_mode,
                p=prompt_bank.p,
                random_suffix_noise_config=random_suffix_noise_config,
            )
        )
    return shared_build_dataset_meta(
        prompt_bank=prompt_bank,
        prompt_bank_dir=prompt_bank_dir,
        teacher_checkpoint=teacher_checkpoint,
        subset_size=subset_size,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode="tokens",
        gen_batch_size=gen_batch_size,
        device=device,
        dtype_name=dtype_name,
        seed=seed,
        extra_meta=extra_meta,
    )


def render_offline_dataset(
    *,
    teacher_checkpoint: str | Path,
    prompt_bank_dir: str | Path,
    save_dir: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
    teacher_law: str = "distributional_noise",
    random_suffix_noise_config=None,
) -> None:
    # Rendered datasets are the fixed D_eta trajectories in the paper's
    # LogLossBC baseline. The clean validation split is copied from the prompt
    # bank so all methods report comparable clean-task metrics.
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    prompt_bank = load_prompt_bank(prompt_bank_dir)
    subset_idx = select_train_subset(prompt_bank, subset_size)
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        model = load_native_teacher(
            teacher_checkpoint,
            device=device,
            dtype_name=dtype_name,
        )
        train_x, train_y, train_teacher_probs = _render_random_suffix_train_split(
            model,
            prompt_bank,
            subset_idx,
            eta=eta,
            rollout_mode=rollout_mode,
            gen_batch_size=gen_batch_size,
            device=device,
            random_suffix_noise_config=random_suffix_noise_config,
        )
        val_x, val_y = build_oracle_val_split(prompt_bank)
        meta = build_dataset_meta(
            prompt_bank=prompt_bank,
            prompt_bank_dir=prompt_bank_dir,
            teacher_checkpoint=teacher_checkpoint,
            subset_size=subset_size,
            eta=eta,
            rollout_mode=rollout_mode,
            gen_batch_size=gen_batch_size,
            device=device,
            dtype_name=dtype_name,
            seed=seed,
            teacher_law=teacher_law,
            random_suffix_noise_config=random_suffix_noise_config,
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
        return
    shared_render_offline_dataset(
        teacher_checkpoint=teacher_checkpoint,
        prompt_bank_dir=prompt_bank_dir,
        save_dir=save_dir,
        subset_size=subset_size,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode="tokens",
        gen_batch_size=gen_batch_size,
        device=device,
        dtype_name=dtype_name,
        seed=seed,
        prompt_bank=prompt_bank,
        subset_idx=subset_idx,
        corrupt_ids_fn=lambda ids, noise: corrupt_ids(ids, noise, p=prompt_bank.p),
        extra_meta={"teacher_law": teacher_law},
    )


def _render_random_suffix_train_split(
    model,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    *,
    eta: float,
    rollout_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    random_suffix_noise_config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
    validate_random_suffix_applies_to_task(config, task_name="modadd")
    subset_size = int(subset_idx.numel())
    target_len = prompt_bank.target_len
    train_x = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.token_dtype)
    train_y = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.label_dtype)
    generator = make_random_suffix_generator(device=device, seed=config.seed)

    for start in range(0, subset_size, gen_batch_size):
        end = min(start + gen_batch_size, subset_size)
        batch_idx = subset_idx[start:end]
        batch_prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, batch_idx)
        batch_target_ids, _, _ = _generate_random_suffix_targets(
            model,
            batch_prompt_ids,
            target_len=target_len,
            eta=eta,
            rollout_mode=rollout_mode,
            device=device,
            p=prompt_bank.p,
            random_suffix_noise_config=config,
            generator=generator,
        )
        batch_x, batch_y = build_xy_from_prompt_and_target(batch_prompt_ids, batch_target_ids)
        train_x[start:end] = batch_x
        train_y[start:end] = batch_y
        print(f"train: rendered {end}/{subset_size}")

    return train_x, train_y, None
